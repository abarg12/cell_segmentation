"""
helper functions used in all segmentation scripts
"""

import os
import numpy as np
import cv2
from skimage.measure import regionprops


# loads images in 3 color formats
def load_image(image_dir):
    img_folder = os.path.join(image_dir, "images")
    fname = [f for f in os.listdir(img_folder)][0]
    # The cv2 library loads images in BGR format but our segmentation
    # functions require RGB and grayscale images.
    img_bgr  = cv2.imread(os.path.join(img_folder, fname))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_rgb, img_gray


# creates 2D binary masks of the BBBC038 nuclei location solutions which are
# encoded originally with run-length encoding 
def load_ground_truth(image_id, solution_df, height, width):
    gt_masks = []
    for _, row in solution_df[solution_df["ImageId"] == image_id].iterrows():
        mask = np.zeros(height * width, dtype=bool)
        tokens = [int(x) for x in row["EncodedPixels"].split()]
        for i in range(0, len(tokens), 2):
            start = tokens[i] - 1 # converts from 1-based to 0-based index
            length = tokens[i + 1]
            mask[start: start + length] = True
        gt_masks.append(mask.reshape((width, height)).T)
    return gt_masks


# Intersection over Union evaluation metric
def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0


# Dice evaluation metric
def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 2 * inter / denom if denom > 0 else 0.0


# Matches ground truth nucleus locations with predicted ones.
# For it to be a match, the IoU must be > 0.5
def evaluate_predictions(pred_labels, gt_masks, iou_thresh=0.5):
    pred_ids = [r.label for r in regionprops(pred_labels)]
    if not pred_ids or not gt_masks:
        return 0.0, 0.0
 
    ious = []
    dices = []
    used = set()  # tracks which predictions have already been matched
 
    for gt in gt_masks:
        best_iou = 0.0
        best_dice = 0.0
        best_id = None
 
        for pid in pred_ids:
            if pid in used:
                continue
            pmask = pred_labels == pid
            score = iou(pmask, gt)
            if score > best_iou:
                best_iou = score
                best_dice = dice(pmask, gt)
                best_id = pid
 
        # only claim the prediction if it overlaps enough to count as a match
        if best_id is not None and best_iou >= iou_thresh:
            used.add(best_id)
 
        ious.append(best_iou)
        dices.append(best_dice)
 
    return np.mean(ious), np.mean(dices)

