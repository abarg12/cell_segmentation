# Cell Nucleus Segmentation

Comparison of three segmentation methods for cell nuclei using
images from the BBBC038 (2018 Kaggle Data Science Bowl) dataset.

- Watershed: marker-based watershed (`watershed.py`), implemented from scratch
- SLIC + Otsu: superpixel segmentation merged by Otsu thresholding (`slic_otsu.py`),
  also implemented from scratch
- Cellpose: pretrained `nuclei` deep learning model (`cellpose_baseline.py`),
  used as an off-the-shelf baseline


## Setup
Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

## Data
Download the BBBC038 stage-1 test images and the ground-truth solution CSV:

```bash
python get_data.py
```


## Running the segmentation methods
Each method has its own script. Running with no flags performs a single
default-parameter run on the first 10 test images, prints per-image metrics
(IoU, Dice, count error, runtime), and saves a contour-overlay PNG per image
under `results/<method>/`.

```bash
python watershed.py            # marker-based watershed
python slic_otsu.py            # SLIC + Otsu merging
python cellpose_baseline.py    # Cellpose pretrained 'nuclei' model
```

**First Cellpose run.** The pretrained model weights are downloaded
automatically the first time `cellpose_baseline.py` runs. Subsequent runs 
usethe cached copy.

## Test mode (parameter sweeps)

`watershed.py` and `slic_otsu.py` have a `--test` flag (or `-t`) that
sweeps two key parameters per method, evaluates each setting on the same
10-image test set, and saves plots in `plots/<method>/`.

```bash
python watershed.py --test     # sweeps blur_sigma and min_distance
python slic_otsu.py --test     # sweeps n_segments and compactness
```

`cellpose_baseline.py` does not have a test mode.

