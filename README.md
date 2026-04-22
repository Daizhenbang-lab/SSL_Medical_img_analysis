# SSL Medical Image Analysis

This project implements a semi-supervised self-supervised learning workflow for medical and histopathology images. The pipeline extracts tissue/sample regions and masks from whole-slide images, cuts image patches, fine-tunes a SimCLR model with optional regression supervision, generates feature embeddings and 3D UMAP coordinates, and projects patch-level colors or clusters back into image space.

## Project Structure

```text
.
|-- data_helper.py                     # Dataset wrapper for labeled and unlabeled patches
|-- extract_patch.py                   # Patch extraction from sample images and masks
|-- feature_extract.py                 # Feature extraction, UMAP generation, and h5ad export
|-- get_full_sample&mask.py            # Full sample extraction from WSI images and masks
|-- patch_project.py                   # Projection of UMAP or cluster colors back to image space
|-- Sample_split.py                    # Experimental spatial split utility based on skeleton lines
|-- semisupervised_finetune_simclr.py  # Semi-supervised SimCLR LightningModule
|-- trainer.py                         # Training entry point
|-- visualize_cluster.py               # DBSCAN cluster visualization
|-- utils/                             # Image preprocessing, patch extraction, model, and embedding utilities
`-- stainlib/                          # Local stain augmentation and normalization package
```

## Environment Setup

Python 3.9-3.11 is recommended. Create and activate a virtual environment before installing the dependencies.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version first, then install the rest of the dependencies. Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The repository includes a local copy of `stainlib`. It is installed in editable mode through `-e ./stainlib` in `requirements.txt`.

## Data Layout

The scripts use the following directories by default. You can override most paths with command-line arguments.

```text
WSI_PSA/               # Raw whole-slide or large input images
full_mask_PSA/         # Tissue/sample masks
mask_PSA_Positive/     # Positive-region masks
full_sample_PSA/       # Extracted sample-region images
dataset_size56_PSA/    # Extracted patch dataset output
```

The training code also expects an initialization checkpoint by default:

```text
tenpercent_resnet18.ckpt
```

Place this file in the project root, or update `path_to_model` in the code.

## Typical Workflow

### 1. Extract Full Sample Regions

```bash
python "get_full_sample&mask.py" ^
  --slide_path WSI_PSA ^
  --save_sample_path full_sample_PSA ^
  --save_mask_path full_mask_PSA
```

The current script reads an existing grayscale mask from `save_mask_path` and uses it to generate the sample-region image. If you want to generate masks directly from input images, enable the `Mask_Extraction(slide)` logic in the script.

### 2. Extract Patches

```bash
python extract_patch.py ^
  --slide_path full_sample_PSA ^
  --mask_path full_mask_PSA ^
  --positive_mask_path mask_PSA_Positive ^
  --patch_shape 56 ^
  --overlap 0.5 ^
  --save_path dataset_size56_PSA ^
  --mask_th 0.6
```

Each sample folder receives patch images and a matching `coords_*.csv` file. Patches with a non-empty `positive_pixel` value are treated as labeled samples. Patches with an empty value are used as unlabeled samples for self-supervised training.

### 3. Fine-Tune Semi-Supervised SimCLR

```bash
python trainer.py ^
  --dataset_path dataset_size56_PSA/full_sample_PSA/train ^
  --num_workers 4 ^
  --batch_size 64 ^
  --seed 1 ^
  --max_epoch 50
```

Checkpoints are written to:

```text
checkpoints_method_lambda0.9/
```

Note: `trainer.py` currently contains a hard-coded `csv_path` example list. Update this list to point to your own `coords_*.csv` files before running training on a new dataset.

### 4. Extract Features and Generate UMAP

```bash
python feature_extract.py ^
  --save_path test_set/sample ^
  --architecture resnet18 ^
  --model_path checkpoints_method_lambda0.9/simclr-epoch=46-train_total_loss=15.82.ckpt ^
  --experiment_name test_neighbor2_3D_lambda0.9 ^
  --num_workers 4
```

Main outputs:

```text
embeddings_<experiment_name>.p
UMAP_<experiment_name>.csv
<experiment_name>.h5ad
```

### 5. Visualize Clusters

```bash
python visualize_cluster.py
```

The CSV path and DBSCAN parameters are currently hard-coded examples. Edit them before use:

```python
csv_file_path = 'generated_result/updated_scan167.csv'
visualize_clusters_separately_from_csv(csv_file_path, eps=0.5, min_samples=1250)
```

### 6. Project Patch Results Back to Image Space

```bash
python patch_project.py ^
  --sample_slide WSI_PSA/sample ^
  --umap_file generated_result/UMAP_test_neighbor2_3D_lambda0.9.csv ^
  --save_path test_set/sample ^
  --patch_size 56
```

## Notes

- `trainer.py` currently does not use `--dataset_path` to discover CSV files. Training data still comes from the in-script `csv_path` list.
- `data_helper.py` appends the old absolute path `'/tmp/medical_image/stainlib'`. When running from this repository with `-e ./stainlib` installed, this path should not be required.
- `Sample_split.py`, `visualize_cluster.py`, and `utils/Get_HSV_Value.py` are experimental utility scripts with hard-coded paths. Edit them for your dataset before running.
- Large images, extracted patches, checkpoints, embeddings, and generated UMAP results should usually stay out of Git.

## Quick Check

```bash
python -m compileall -q .
```

This command performs a quick syntax check for the Python files.
