"""
Cellpose deep learning for cell nucleus segmentation.
"""

import os
import time
import numpy as np
import pandas as pd
import cv2
from cellpose import models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import regionprops
from utils import load_image, load_ground_truth, iou, dice, evaluate_predictions


# Runs Cellpose on a grayscale image
def run_cellpose(model, img_gray, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
    t0 = time.time()

    masks, flows, styles = model.eval(
        img_gray,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    elapsed = time.time() - t0
    return masks.astype(np.int32), elapsed


# show predicted cell nuclei as contours on the original image
def visualize(img_bgr, masks, gt_masks=None, title='Cellpose Segmentation', save_path=None):
    orig_image = img_bgr.copy()

    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0] 

    for cell_id in cell_ids:
        mask_u8 = (masks == cell_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orig_image, contours, -1, (0, 255, 0), 1)

    if gt_masks:
        for gm in gt_masks:
            mask_u8 = gm.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(orig_image, contours, -1, (0, 0, 255), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    
    ax2.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(title)

    handles = [mpatches.Patch(color='lime', label='Predicted')]
    if gt_masks:
        handles.append(mpatches.Patch(color='red', label='Ground Truth'))
    ax2.legend(handles=handles, loc='upper right')

    ax1.axis('off')
    ax2.axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved visualization -> {save_path}")
    plt.close()


# Loops over test images, runs the Cellpose model, and then prints and visualizes the results
def main(data_dir="data", num_images=10, use_gpu=False,
         diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
         iou_threshold=0.5, visualize_results=True,
         output_dir="results/cellpose"):
 
    os.makedirs(output_dir, exist_ok=True)
 
    test_dir = os.path.join(data_dir, "stage1_test")
    sol_path = os.path.join(data_dir, "stage1_solution.csv")
    solution_df = pd.read_csv(sol_path)
 
    model = models.CellposeModel(gpu=use_gpu, pretrained_model='nuclei')
 
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
        image_dir = os.path.join(test_dir, image_id)
        img_bgr, _, img_gray = load_image(image_dir)
        h, w = img_gray.shape
 
        # run Cellpose and collect timing data
        masks, elapsed = run_cellpose(model, img_gray, diameter=diameter, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
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
            savePath = os.path.join(output_dir, image_id + "_cellpose.png")
            visualize(img_bgr, masks, gt_masks,
                      title=f"Cellpose IoU={mean_iou:.3f}",
                      save_path=savePath)
 
    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)
    avg_count_err = np.mean(np.abs(all_count_errors))
    avg_time = np.mean(all_times)
 
    print("\n------ Cellpose Summary ------")
    print(f"Mean IoU          : {avg_iou:.4f}")
    print(f"Mean Dice         : {avg_dice:.4f}")
    print(f"Mean |Count Error|: {round(float(avg_count_err), 2)}")
    print(f"Mean Runtime (s)  : {round(float(avg_time), 3)}")


if __name__ == "__main__":
    print("Cellpose Deep Learning Baseline\n")
    print("Note: On first run, Cellpose will download the pretrained model")

    main(
        data_dir="data",
        num_images=10,
        use_gpu=False,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        iou_threshold=0.5,
        visualize_results=True,
        output_dir="results/cellpose",
    )

