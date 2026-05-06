"""
SLIC + Otsu segmentation for cell nuclei
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color
from skimage.segmentation import mark_boundaries
from utils import load_image, load_ground_truth, evaluate_predictions
 
 
# stores cluster center for a superpixel
class SuperPixel:
    def __init__(self, y, x, l=0, a=0, b=0):
        self.update(y, x, l, a, b)
 
    def update(self, y, x, l, a, b):
        self.y = y # row
        self.x = x # column
        self.l = l # lightness
        self.a = a # green-red color axis
        self.b = b # blue-yellow color axis
 
 
# estimates diagonal gradient at pixel (y, x)
def calc_gradient(y, x, img):
    height, width = img.shape[:2]

    # boundary checking, edge pixels re-use neighboring inner pixel values
    y = max(1, min(y, height - 2))
    x = max(1, min(x, width - 2))

    dx = img[y, x+1] - img[y, x-1]
    dy = img[y+1, x] - img[y-1, x]
    return (dx**2).sum() + (dy**2).sum()
 
 
# initialize cluster centers
# S is the pixel spacing between cluster centers
def initial_cluster_centers(S, img):
    height, width = img.shape[:2]

    clusters = []

    # step through image at intervals of S pixels and place cluster centers
    y = S // 2
    while y < height:
        x = S // 2
        while x < width:
            l, a, b = img[y, x]
            clusters.append(SuperPixel(y, x, l, a, b))
            x += S
        y += S
    return clusters
 
 
# move each cluster to the lowest gradient position in its 3x3 neighborhood
def centers_to_min_grad(clusters, img):
    for c in clusters:
        best_grad = calc_gradient(c.y, c.x, img)
        for offset_y in range(-1, 2):
            for offset_x in range(-1, 2):
                new_y = c.y + offset_y
                new_x = c.x + offset_x
                g = calc_gradient(new_y, new_x, img)
                if g < best_grad:
                    l, a, b = img[new_y, new_x]
                    c.update(new_y, new_x, l, a, b)
                    best_grad = g
 
 
# for each cluster center, compute distance to all pixels in its 2S x 2S region
def assign_pixels(clusters, S, img, m, labels, d):
    height, width = img.shape[:2]
    Y, X = np.mgrid[0:height, 0:width]
 
    for k in range(len(clusters)):
        c = clusters[k]
 
        # clamp to image boundaries
        y0 = max(0, c.y - 2*S)
        y1 = min(height, c.y + 2*S)
        x0 = max(0, c.x - 2*S)
        x1 = min(width, c.x + 2*S)
 
        patch = img[y0:y1, x0:x1]
        ys = Y[y0:y1, x0:x1]
        xs = X[y0:y1, x0:x1]
 
        # color distance in CIELAB space
        d_c = np.sqrt((patch[:,:,0] - c.l)**2 +
                      (patch[:,:,1] - c.a)**2 +
                      (patch[:,:,2] - c.b)**2)
 
        # spatial distance
        d_s = ((ys - c.y)**2 + (xs - c.x)**2) ** 0.5
 
        # combined distance measure
        D = ((d_c / m)**2 + (d_s / S)**2) ** 0.5
 
        # update pixel labels and distances when this cluster center is closer
        closer = D < d[y0:y1, x0:x1]
        d[y0:y1, x0:x1][closer] = D[closer]
        labels[y0:y1, x0:x1][closer] = k
 
 
# move each cluster center to the mean [l a b x y] of its assigned pixels
def update_cluster_means(clusters, img, labels):
    height, width = img.shape[:2]
    Y, X = np.mgrid[0:height, 0:width]
 
    for k in range(len(clusters)):
        c = clusters[k]
        mask = labels == k
        if mask.any() == False:
            continue
 
        new_y = int(Y[mask].mean())
        new_x = int(X[mask].mean())
        new_l = float(img[mask, 0].mean())
        new_a = float(img[mask, 1].mean())
        new_b = float(img[mask, 2].mean())
        c.update(new_y, new_x, new_l, new_a, new_b)
 
 
# flood fill to find all connected blobs in a binary mask
# returns a map where each blob gets its own integer id, plus the total blob count
def find_components(binary_mask):
    H, W = binary_mask.shape
    visited = np.zeros((H, W), dtype=bool)
    comp_map = np.zeros((H, W), dtype=np.int32)
    comp_id = 0
 
    for r in range(H):
        for c in range(W):
            if not binary_mask[r, c] or visited[r, c]:
                continue
 
            # new blob found, flood fill from here using a stack
            comp_id += 1
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                if visited[y, x]:
                    continue
                visited[y, x] = True
                comp_map[y, x] = comp_id
                if y > 0 and binary_mask[y-1, x] and not visited[y-1, x]:
                    stack.append((y-1, x))
                if y < H-1 and binary_mask[y+1, x] and not visited[y+1, x]:
                    stack.append((y+1, x))
                if x > 0 and binary_mask[y, x-1] and not visited[y, x-1]:
                    stack.append((y, x-1))
                if x < W-1 and binary_mask[y, x+1] and not visited[y, x+1]:
                    stack.append((y, x+1))
 
    return comp_map
 
 
# SLIC algorithm
#
# n_segments = desired number of superpixels
# compactness = parameter that controls tradeoff between color and spatial distance
def slic(img_rgb, n_segments=200, compactness=10.0):
    height = img_rgb.shape[0]
    width = img_rgb.shape[1]
    num_pixels = height * width
 
    # S is grid spacing between cluster centers
    S = int(np.sqrt(num_pixels / n_segments))
 
    # convert to CIELAB for color distance calculations
    img = color.rgb2lab(img_rgb.astype(np.float64) / 255.0)
 
    # place starting cluster centers on a regular grid
    clusters = initial_cluster_centers(S, img)
    centers_to_min_grad(clusters, img)
 
    # iteratively reassign pixels to their nearest cluster and recompute centers
    for i in range(10):
        labels = np.full((height, width), -1, dtype=np.int32)
        d = np.full((height, width), np.inf, dtype=np.float64)
        assign_pixels(clusters, S, img, compactness, labels, d)
        update_cluster_means(clusters, img, labels)
 
    # renumber labels to remove gaps
    unique_vals = np.unique(labels)
    remap = {}
    for new_id, old_id in enumerate(unique_vals):
        remap[old_id] = new_id
    labels = np.vectorize(remap.get)(labels)
 
    return labels.astype(np.int32)


# perform Otsu, but just return the threshold value
def otsu_get_threshold_val(img_gray):
    rows, cols = img_gray.shape
    num_pixels = rows * cols
    num_levels = 256

    # build histogram by iterating over each pixel and incrementing the bins
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
 
        # Otsu criterion: maximize between-class variance
        between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
 
        if between_class_var > best_var:
            best_var = between_class_var
            best_threshold = threshold
 
    return best_threshold

 
# Run SLIC, then label each superpixel as cell-or-background based on its
# mean grayscale intensity using an Otsu-chosen threshold. Connected
# components of the cell-labeled superpixels become candidate cell instances.
def segment(img_rgb, img_gray, n_segments=200, compactness=10.0, min_cell_area=100):
    superpixels = slic(img_rgb, n_segments, compactness)
 
    # find the mean grayscale per superpixel computed with bincount
    flat_sp = superpixels.ravel()
    flat_gray = img_gray.astype(np.float64).ravel()

    # sums up intensity values for each superpixel
    intensity_sums = np.bincount(flat_sp, weights=flat_gray)

    # creates data structure with number of pixels in each superpixel
    pixels_in_sp = np.bincount(flat_sp)
    sp_means = intensity_sums / np.maximum(pixels_in_sp, 1)
 
    # Otsu threshold computed on the superpixel means rather than raw pixels
    threshold = otsu_get_threshold_val(img_gray)
 
    # Decide which side of the threshold is the nucleus color
    # Makes assumption that there are always less nucleus superpixels 
    pixels_below = int((img_gray < threshold).sum())
    pixels_above = img_gray.size - pixels_below
    nuclei_are_dark = pixels_below < pixels_above
 
    if nuclei_are_dark:
        is_cell = sp_means < threshold
    else:
        is_cell = sp_means > threshold
 
    # paint a binary foreground mask via per-superpixel lookup
    foreground = is_cell[superpixels]
 
    # connected components of the foreground are chosen as candidate cells
    component_map = find_components(foreground)
 
    # drop segments smaller than min_cell_area, then renumber
    instance_labels = np.zeros_like(component_map, dtype=np.int32)
    next_label = 1
    for component_id in np.unique(component_map):
        # background components are labeled with 0
        if component_id == 0:
            continue
        mask = component_map == component_id

        # filter out segments that we say aren't big enough to be cells
        if mask.sum() >= min_cell_area:
            instance_labels[mask] = next_label
            next_label += 1
 
    return superpixels, instance_labels
 
 
def visualize(img_bgr, img_rgb, superpixels, instance_labels,
              gt_masks=None, title="SLIC", save_path=None):
    orig_image = img_bgr.copy()

    cell_ids = np.unique(instance_labels)
    cell_ids = cell_ids[cell_ids != 0]

    # draw predicted nucleus boundaries in green
    for cell_id in cell_ids:
        mask_u8 = (instance_labels == cell_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orig_image, contours, -1, (0, 255, 0), 1)

    # if ground truth masks exist plot the contours in red
    if gt_masks:
        for gm in gt_masks:
            mask_u8 = gm.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(orig_image, contours, -1, (0, 0, 255), 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))

    # left panel is the original image
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image')

    # middle panel has the SLIC superpixel grid drawn on
    n_sp = len(np.unique(superpixels))
    ax2.imshow(mark_boundaries(img_rgb, superpixels, color=(1, 0.5, 0)))
    ax2.set_title("SLIC superpixels (n~" + str(n_sp) + ")")

    # right panel has predicted and ground truth contours plotted
    ax3.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    ax3.set_title(title)

    handles = [mpatches.Patch(color='lime', label='Predicted')]
    if gt_masks:
        handles.append(mpatches.Patch(color='red', label='Ground Truth'))
    ax3.legend(handles=handles, loc='upper right')

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.tight_layout()
    # save to disk if a path was given
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved visualization -> {save_path}")
    plt.close()
 
 
def main(data_dir="data", num_images=10, n_segments=200,
         compactness=10.0, min_cell_area=100, iou_thresh=0.5,
         visualize_results=True, output_dir="results/slic_otsu",
         test_mode=False):
    os.makedirs(output_dir, exist_ok=True)
    solution_df = pd.read_csv(os.path.join(data_dir, "stage1_solution.csv"))
    test_dir = os.path.join(data_dir, "stage1_test")
    image_ids = sorted(os.listdir(test_dir))[:num_images]
 
    all_ious = []
    all_dices = []
    all_count_errors = []
    all_times = []
 
    img_num = 1
    for image_id in image_ids:
        img_bgr, img_rgb, img_gray = load_image(os.path.join(test_dir, image_id))
        h, w = img_gray.shape
 
        t0 = time.time()
        superpixels, instance_labels = segment(
            img_rgb, img_gray, n_segments=n_segments,
            compactness=compactness, min_cell_area=min_cell_area)
        elapsed = time.time() - t0
        all_times.append(elapsed)
 
        gt_masks = load_ground_truth(image_id, solution_df, h, w)
        mean_iou, mean_dice = evaluate_predictions(instance_labels, gt_masks, iou_thresh)
        all_ious.append(mean_iou)
        all_dices.append(mean_dice)
 
        num_preds = len(np.unique(instance_labels)) - 1
        gt_count = len(gt_masks)
        count_err = num_preds - gt_count
        all_count_errors.append(count_err)
 
        if not test_mode:
            print("")
            print(f"Image {img_num}")
            print(f"Image ID: {image_id}")
            print(f"  IoU = {round(mean_iou, 3)}")
            print(f"  Dice = {round(mean_dice, 3)}")
            print(f"  Num Predictions = {num_preds}")
            print(f"  Num Ground Truth = {gt_count}")
            print(f"  Count Error = {count_err}")
            print(f"  Runtime = {round(elapsed, 3)} s")
            print("")
 
        img_num += 1
 
        if visualize_results:
            save_path = os.path.join(output_dir, image_id + "_slic_otsu.png")
            visualize(img_bgr, img_rgb, superpixels, instance_labels, gt_masks,
                      title=f"SLIC+Otsu  IoU={mean_iou:.3f}",
                      save_path=save_path)
 
    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)
    avg_count_err = np.mean(np.abs(all_count_errors))
    avg_time = np.mean(all_times)
 
    print("--- SLIC + Otsu summary ---")
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
        # This test varies the number of superpixels that SLIC is
        # initialized with (n_segments)
        for val in [50, 100, 200, 400]:
            print(f"Testing n_segments = {val}")
            main(
                data_dir="data",
                num_images=10,
                n_segments=val,
                compactness=10.0,
                min_cell_area=100,
                iou_thresh=0.5,
                visualize_results=False,
                test_mode=True)

        # This test varies the compactness parameter which affects how
        # spatial proximity is weighted against color similarity
        for val in [5.0, 10.0, 20.0, 40.0]:
            print(f"Testing compactness = {val}")
            main(
                data_dir="data",
                num_images=10,
                n_segments=200,
                compactness=val,
                min_cell_area=100,
                iou_thresh=0.5,
                visualize_results=False,
                test_mode=True)

    # Run SLIC with the default parameters once per image
    else:
        main(
            data_dir="data",
            num_images=10,
            n_segments=200,
            compactness=10.0,
            min_cell_area=100,
            iou_thresh=0.5,
            visualize_results=True,
            output_dir="results/slic_otsu")