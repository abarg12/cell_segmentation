"""
Marker-controlled watershed segmentation for cell nucleus images
"""

import argparse
import os
import time
import heapq
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           label as scipy_label)
from utils import load_image, load_ground_truth, evaluate_predictions



# perform Otsu thresholding binarize image into foreground and background
def otsu_threshold(img_gray):
    rows, cols = img_gray.shape
    num_pixels = rows * cols
    num_levels = 256

    # build histogram by iterating over the pixels and incrementing the bins
    hist = np.zeros(num_levels)
    for r in range(rows):
        for c in range(cols):
            hist[img_gray[r, c]] += 1

    # normalize probability distribution
    prob = hist / num_pixels
 
    best_var = 0
    best_threshold = 0
 
    for threshold in range(1, num_levels - 1):
        # fraction of pixels in each class
        weight_bg = sum(prob[0:threshold+1])
        weight_fg = sum(prob[threshold+1:num_levels])
 
        if weight_bg == 0 or weight_fg == 0:
            continue
 
        # mean gray level of each class
        mean_bg = sum(level * prob[level] for level in range(0, threshold+1)) / weight_bg
        mean_fg = sum(level * prob[level] for level in range(threshold+1, num_levels)) / weight_fg
 
        # maximize between-class variance
        between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
 
        if between_class_var > best_var:
            best_var = between_class_var
            best_threshold = threshold
 
    binary = img_gray > best_threshold
    # invert if foreground is the majority, dark nuclei on bright background
    if binary.sum() > binary.size * 0.5:
        binary = ~binary
    return binary


# checks neighbors to see if pixel is local maximum
#
# For each pixel, look at the (2*min_distance+1) by (2*min_distance+1) square
# window centered on it. A pixel qualifies as a local maximum if it has
# the largest value in that window
def find_local_maxima(image, min_distance=10):
    H, W = image.shape
    half = min_distance

    local_max = np.zeros((H, W), dtype=bool)

    for y in range(H):
        for x in range(W):
            # background pixels can't be markers
            if image[y, x] <= 0:
                continue

            # clamp the (2*half+1) by (2*half+1) window to the image bounds
            y0 = max(0, y - half)
            y1 = min(H, y + half + 1)
            x0 = max(0, x - half)
            x1 = min(W, x + half + 1)

            # find the maximum value inside the window
            window_max = image[y0, x0]
            for wy in range(y0, y1):
                for wx in range(x0, x1):
                    if image[wy, wx] > window_max:
                        window_max = image[wy, wx]

            # this pixel is a local max if its value equals the window max
            if image[y, x] == window_max:
                local_max[y, x] = True

    return local_max


# Use Meyer's flooding algorithm to segment the image.
# Requires markers to be passed in as a parameter to start the flooding.
def watershed(image, markers, mask=None):
    H, W = image.shape

    # labels will hold the result of flooding
    labels = markers.copy()

    # inQueue keeps track of which pixels we have already added to queue
    inQueue = np.zeros((H, W), dtype=bool)

    # keeps track of the pixels we are actively checking
    priorityQueue = []

    neighbors = [(-1,0), (1,0), (0,-1), (0,1)]

    # add unlabeled pixels to queue if they are adjacent to markers
    for y in range(H):
        for x in range(W):
            if labels[y, x] <= 0:
                continue
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if inQueue[ny, nx] or labels[ny, nx] > 0:
                    continue
                if mask is not None and not mask[ny, nx]:
                    continue
                heapq.heappush(priorityQueue, (float(image[ny, nx]), ny, nx))
                inQueue[ny, nx] = True

    # continue iterating to flood 
    while len(priorityQueue) > 0:
        _, y, x = heapq.heappop(priorityQueue)

        neighborLabels = set()
        for dy, dx in neighbors:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] > 0:
                neighborLabels.add(labels[ny, nx])

        if len(neighborLabels) == 1:
            labels[y, x] = neighborLabels.pop()
        elif len(neighborLabels) > 1:
            labels[y, x] = -1 # boundary pixel

        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if inQueue[ny, nx] or labels[ny, nx] > 0:
                continue
            if mask is not None and not mask[ny, nx]:
                continue
            heapq.heappush(priorityQueue, (float(image[ny, nx]), ny, nx))
            inQueue[ny, nx] = True

    return labels


# The full marker-based Watershed segmentation pipeline
# 
# We pre-process by applying a Gaussian filter, thresolding to a binary image,
# and then using a distance transform to find initial markers for Watershed.
def segment(img_gray, blur_sigma=2.0, min_distance=10):
    smoothed = gaussian_filter(img_gray.astype(np.float64), sigma=blur_sigma)
    binary = otsu_threshold(smoothed.astype(np.uint8))
    dist = distance_transform_edt(binary)

    maxima_mask = find_local_maxima(dist, min_distance=min_distance)
    markers, _ = scipy_label(maxima_mask)

    # negative distance so cell centers are valleys for the flooding step
    labels = watershed(-dist, markers, mask=binary)
    return labels


# print out the image results as contours over the original image
def visualize(img_bgr, masks, gt_masks=None, title="Watershed", save_path=None):
    orig_image = img_bgr.copy()

    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0]

    # draw each predicted nucleus boundary in green
    for cell_id in cell_ids:
        mask_u8 = (masks == cell_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orig_image, contours, -1, (0, 255, 0), 1)

    # if ground truth masks exist plot the contours in red
    if gt_masks:
        for gm in gt_masks:
            mask_u8 = gm.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(orig_image, contours, -1, (0, 0, 255), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # left figure is the original image
    ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')

    # right figure has predicted and ground truth contours plotted
    ax2.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(title)

    handles = [mpatches.Patch(color='lime', label='Predicted')]
    if gt_masks:
        handles.append(mpatches.Patch(color='red', label='Ground Truth'))
    ax2.legend(handles=handles, loc='upper right')

    ax1.axis('off')
    ax2.axis('off')

    plt.tight_layout()
    # save to disk if a path was given
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved visualization -> {save_path}")
    plt.close()


# Loops over test images, runs Watershed, and then prints and visualizes the results
def main(data_dir="data", num_images=10, blur_sigma=2.0,
        min_distance=10, iou_threshold=0.5,
        visualize_results=True, output_dir="results/watershed",
        test_mode=False):
    os.makedirs(output_dir, exist_ok=True)
 
    test_dir = os.path.join(data_dir, "stage1_test")
    sol_path = os.path.join(data_dir, "stage1_solution.csv")
    solution_df = pd.read_csv(sol_path)
 
    # get the first num_images image IDs from the test folder
    all_ids = os.listdir(test_dir)
    all_ids = sorted(all_ids)
    image_ids = []
    for i in range(num_images):
        image_ids.append(all_ids[i])
 
    all_ious = []
    all_dices = []
    all_count_errors = []
    all_times = []

    img_num = 1
    for image_id in image_ids:
        img_bgr, _, img_gray = load_image(os.path.join(test_dir, image_id))
        h, w = img_gray.shape

        # the segment function does pre-processing steps and calls the main Watershed algorithm
        t0 = time.time()
        masks = segment(img_gray, blur_sigma=blur_sigma, min_distance=min_distance)
        elapsed = time.time() - t0
        all_times.append(elapsed)

        # collect the segmentation error metrics (IoU and Dice)
        gt_masks = load_ground_truth(image_id, solution_df, h, w)
        mean_iou, mean_dice = evaluate_predictions(masks, gt_masks, iou_threshold)
        all_ious.append(mean_iou)
        all_dices.append(mean_dice)

        # compare how many cells we counted compared to base truth
        numPreds = len(np.unique(masks)) - 1  # subtract background label (0)
        gt_count = len(gt_masks)
        count_err = numPreds - gt_count
        all_count_errors.append(count_err)

        if not test_mode:
            print("")
            print(f"Image {img_num}")
            print(f"Image ID: {image_id}")
            print(f"  IoU = {round(mean_iou, 3)}")
            print(f"  Dice = {round(mean_dice, 3)}")
            print(f"  Num Predictions = {numPreds}")
            print(f"  Num Ground Truth = {gt_count}")
            print(f"  Count Error = {count_err}")
            print(f"  Runtime = {round(elapsed, 3)} s")
            print("")

        img_num += 1

        if visualize_results:
            savePath = os.path.join(output_dir, image_id + "_watershed.png")
            visualize(img_bgr, masks, gt_masks,
                      title=f"Watershed IoU={mean_iou:.3f}",
                      save_path=savePath)
    
    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)
    avg_count_err = np.mean(np.abs(all_count_errors))
    avg_time = np.mean(all_times)

    print("------ Watershed Summary ------")
    print(f"Mean IoU          : {avg_iou:.4f}")
    print(f"Mean Dice         : {avg_dice:.4f}")
    print(f"Mean |Count Error|: {round(float(avg_count_err), 2)}")
    print(f"Mean Runtime (s)  : {round(float(avg_time), 3)}\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    # Adds option to evaluate variety of SLIC parameters.
    # Running in this mode will test the effects of varying
    # n_segments and compactness.
    arg_parser.add_argument("-t", "--test", action="store_true")
    args = arg_parser.parse_args()

    if args.test:
        # Tests how varying the smoothing (blur_sigma) affects
        # the performance of Watershed segmentation
        for val in [1.0, 2.0, 4.0, 6.0]:
            print(f"Testing blur_sigma = {val}")
            main(
                data_dir="data",
                num_images=10,
                blur_sigma=val,
                min_distance=10,
                iou_threshold=0.5,
                visualize_results=False,
                test_mode=True
            )

        # Tests how varying the min_distance affects the
        # performance of Watershed segmentation
        for val in [5, 10, 15, 20]:
            print(f"Testing min_distance = {val}")
            main(
                data_dir="data",
                num_images=10,
                blur_sigma=2.0,
                min_distance=val,
                iou_threshold=0.5,
                visualize_results=False,
                test_mode=True
            )


    # Runs Watershed with default parameters once on each image
    else:
        main(
            data_dir="data",
            num_images=10,
            blur_sigma=2.0,
            min_distance=10,
            iou_threshold=0.5,
            visualize_results=True,
            output_dir="results/watershed",
        )