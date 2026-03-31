# MATL: Multi-Annotation Triplet Loss

This repository implements the Multi-Annotation Triplet Loss (MATL) method, as described in the [paper](https://arxiv.org/abs/2504.08054) accepted for Oral Presentation at [IGARSS 2025](https://www.2025.ieeeigarss.org/index.php).

**Dataset:**  
[AWIR dataset](https://projectportal.gri.msstate.edu/awir/) collected by Mississippi State Geosystems Research Institute.

Managed by [Meilun Zhou](https://www.linkedin.com/in/meilun-zhou/) through [GatorSense](https://faculty.eng.ufl.edu/machine-learning/).

---

## Environment Setup

You can build the environment using either Conda or pip.

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate mmctl
```

### Using pip
```bash
pip install -r requirements.txt
```

---

## Python Files Overview

- **model.py**  
	Contains the core model architectures, including:
	- Encoder and decoder networks for image and feature processing.
	- Functions for image reconstruction (with and without triplet loss).
	- Classifier architectures for 3-class and 9-class problems.
	- Utilities for building and summarizing Keras models.

- **multi_annotation_triplet_loss.py**  
	Implements the triplet loss functions with online triplet mining:
	- Computes pairwise distances and valid triplet masks.
	- Defines the main triplet loss and a double-loss variant for multi-task learning (classification and bounding box aspect ratio).

- **preprocessing.py**  
	Data preprocessing utilities for the AWIR dataset:
	- Extracts bounding boxes from annotation files.
	- Creates Pandas DataFrames from image folders and annotations.
	- Functions for plotting images with bounding boxes and filtering images by size.


