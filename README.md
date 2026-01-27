# WeedOut: Autonomous Weed Management using Semi-Supervised Non-CNN Annotation

<div align="center">

[![Project Page](https://img.shields.io/badge/Project%20Page-kirankkethineni.github.io/WeedOut-brightgreen?style=for-the-badge)](https://kirankkethineni.github.io/WeedOut/)
[![Paper](https://img.shields.io/badge/Paper-IFIP--IoT%202023-blue?style=for-the-badge)](https://ifipiot.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)](LICENSE)

**[Project Page](https://kirankkethineni.github.io/WeedOut/) | [Paper (IFIP-IoT 2023)](#citation) | [Notebook](WeedOut.ipynb)**

*An Agriculture Cyber-Physical System (A-CPS) that identifies and suppresses weeds without any prior training data — using only crop shape and a single farmer interaction.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Results](#results)
- [Comparison with Prior Work](#comparison-with-prior-work)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Authors](#authors)

---

## Overview

**WeedOut** is a weed management system published at the **IFIP International Internet of Things Conference (IFIP-IoT), 2023**. It is a complete **Agriculture Cyber-Physical System (A-CPS)** — from a drone that captures field images, to an algorithm that clusters crops by shape, to a farmer confirming which cluster is the primary crop, to an autonomous ground rover that applies herbicide precisely at weed locations.

### Key Highlights

| Feature | Detail |
|---|---|
| **No training data needed** | Zero labeled images required — works out of the box |
| **One farmer interaction** | Farmer simply selects which cluster is their crop |
| **Generalizes everywhere** | Works across crop types, growth stages, and geographies |
| **Computationally lightweight** | Only two passes through each image — runs on a phone/tablet |
| **Weed pressure reporting** | Tells farmer what weed types are present and their percentage |
| **End-to-end automation** | Drone → Algorithm → Farmer → Autonomous sprayer |

---

## The Problem

Modern CNN-based weed detection systems suffer from fundamental limitations:

- **Require massive labeled datasets** — thousands of images per crop type, per growth stage, per region.
- **Don't generalize** — a model trained on Iowa corn at 3 weeks cannot reliably classify Nebraska corn at 6 weeks.
- **Fail on unknown weeds** — if a new weed species appears, the CNN has never seen it and cannot classify it.
- **Geography-locked** — the same crop species looks visually different under different lighting, soil, and climate conditions.

The result: farmers either spend enormous resources collecting and labeling training data, or deploy a system that fails when conditions differ from training time.

---

## The Solution

WeedOut takes a fundamentally different approach: **classify crops by shape, not appearance**.

A corn plant looks like a corn plant — tall with wide leaves — whether it grows in Texas or Wisconsin. A weed growing among corn has a *different structural shape*. WeedOut captures this structural identity through **Profile Plots** — a 1D signal of how a plant's width changes along its height — and clusters plants with similar shapes together using **Dynamic Time Warping (DTW)**.

The farmer then sees one image from each cluster and answers a single question: *"Which of these is your primary crop?"* Everything else is automatic.

### Why This Works
- **Shape is invariant** to lighting, color, soil type, and geographic location.
- **No training phase** — clustering is unsupervised; only the final label step needs the farmer.
- **Unknown weeds are handled naturally** — any plant that doesn't match the primary crop cluster is labeled a weed, regardless of species.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WeedOut A-CPS Pipeline                             │
│                                                                             │
│  ┌──────────┐    ┌────────────────────┐    ┌───────────────────────────┐   │
│  │  Drone / │    │  Processing Unit   │    │  Farmer Interface         │   │
│  │  Rover   │───▶│  (Phone/Tablet)    │───▶│  (Select Primary Crop)   │   │
│  │  Captures│    │  Runs WeedOut Algo │    │                           │   │
│  │  Images  │    │                    │    └─────────────┬─────────────┘   │
│  └──────────┘    └────────────────────┘                  │                 │
│                                                          ▼                 │
│                       ┌─────────────────────────────────────────────┐     │
│                       │  Annotated Field Image                      │     │
│                       │  Green = Primary Crop | Red = Weed         │     │
│                       └──────────────────────┬──────────────────────┘     │
│                                              │                             │
│                                              ▼                             │
│                       ┌─────────────────────────────────────────────┐     │
│                       │  Autonomous Weed Sprayer (Ground Rover)     │     │
│                       │  Sprays herbicide at every red pixel        │     │
│                       └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Step 1 — Background Removal

The input field image is converted to **HSV color space** and a green-range threshold is applied to isolate vegetation (crops and weeds) from soil. The image is resized to **250×250 pixels** for computational efficiency.

```
Input Image → HSV Conversion → Green Thresholding → Binary Mask
```

### Step 2 — Individual Crop Identification (Connected Component Labeling)

A **Two-Pass Connected Component Labeling** algorithm scans the binary image and assigns a unique label to each distinct vegetation object (each individual plant). The two passes are:

- **Pass 1:** Scan pixel by pixel. If a pixel's neighbors already have labels, inherit the minimum label and record equivalences. Otherwise, assign a new unique label.
- **Pass 2:** Resolve all equivalences so every connected object has exactly one unique label.

```
Binary Image → Two-Pass CCL → Each plant uniquely labeled
```

### Step 3 — Profile Plot Generation

For each labeled crop, a **Profile Plot** is computed — a 1D array where each element represents the width of the crop (number of active pixels) at each row of its bounding box, from top to bottom. This encodes the plant's structural silhouette.

All profile plots are interpolated to a uniform length of **250 points** for consistent comparison across plants of different sizes.

```
Labeled Crop → Row-wise pixel widths → 250-point Profile Plot
```

### Step 4 — Shape-Based Clustering (Dynamic Time Warping)

**Dynamic Time Warping (DTW)** computes pairwise distances between all profile plots. DTW is ideal for this task because it handles plants at different scales and orientations by finding the optimal non-linear alignment between two 1D signals.

Crops with low DTW distance (similar shapes) are grouped together. Clusters sharing overlapping members are iteratively merged until convergence.

```
Profile Plots → DTW Distance Matrix → Cluster Merging → Distinct Crop Type Clusters
```

### Step 5 — Semi-Supervised Labeling (One Farmer Interaction)

The farmer is shown one representative sample image from each cluster, along with the cluster's percentage of total vegetation area. The farmer selects which cluster is the primary crop. All other clusters are automatically labeled as **weeds**.

```
Clusters → Farmer selects primary crop → Primary = Green, Rest = Red (Weeds)
```

### Step 6 — Weed Pressure Calculation & Sprayer Output

- **Weed pressure** = (total weed area) / (total vegetation area) × 100%
- The annotated image (green/red pixels) is scaled to real farm dimensions.
- The autonomous sprayer rover reads the image pixel by pixel and activates the herbicide nozzle wherever a red (weed) pixel appears.

```
Annotated Image → Scale to farm dimensions → Sprayer applies herbicide at red pixels
```

---

## Results

Evaluated on **20 combined farmland images** from the [Kaggle Crop and Weed Detection Dataset](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes):

| Metric | Value |
|---|---|
| Clustering Accuracy | **93%** |
| F1 Score — Primary Crop | **0.80** |
| F1 Score — Weeds | **0.95** |

The high weed F1 score (0.95) is especially significant — precision in weed identification directly determines herbicide targeting accuracy, minimizing chemical waste and crop damage.

### Single Image Example

On a sample field image with 8 individual plants:
1. Thresholding + Connected Component Labeling → 8 plants identified
2. DTW Clustering → 3 distinct shape clusters found
3. Farmer selects Cluster 1 as primary crop
4. Clusters 2 & 3 labeled as weeds (colored red)
5. Weed pressure computed and displayed

---

## Comparison with Prior Work

| System | Assumption Required | Features Used | Limitation |
|---|---|---|---|
| Louargant et al. (2019) | Row cultivation | Spatial + spectral | Specific to crops varying in vegetation indices |
| Ota et al. (2022) | Row cultivation | Spatial + geometric | Needs more weeds than primary crops |
| Bah et al. (2017) | Row cultivation | Position + superpixel | Row-cultivated crops only |
| Rani et al. (2017) | Crop area > weed area | Pixel area | Oversized weeds misclassified |
| Aravind et al. (2015) | Weed area > crop area | Pixel area | Undersized weeds misclassified |
| Persson & Astrand (2008) | None | Shape (ASM) | Requires training with crop shapes |
| **WeedOut (2023)** | **None** | **Shape (profile plots)** | **No training needed; universal** |

---

## Getting Started

### Prerequisites

```bash
pip install opencv-python numpy matplotlib scipy pandas scikit-learn
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/kirankkethineni/WeedOut.git
   cd WeedOut
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook WeedOut.ipynb
   ```

3. Place your field image as `finalc.jpg` in the project root (or update the path in the notebook).

4. Run all cells. At the clustering step, review the displayed cluster samples and identify which cluster is your primary crop.

### Dataset

The experiments used the [Kaggle Crop and Weed Detection Dataset](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes) by Ravirajsinh Dabhi and Dhruv Makwana (2020).

---

## Repository Structure

```
WeedOut/
├── WeedOut.ipynb       # Main notebook — complete pipeline implementation
├── finalc.jpg          # Sample field image for testing
└── README.md           # This file
```

---

## Limitations & Future Work

**Current Limitations:**
- Connected Component Labeling counts overlapping plants as a single entity — two touching plants get one label.
- If two different species have very similar profile shapes, they may be clustered together.

**Future Directions:**
- Incorporate **edge detection** to separate overlapping crops before labeling.
- Add **additional geometric features** (leaf count, perimeter, aspect ratio) alongside profile plots to handle ambiguous shape cases.
- Extend to **multi-crop fields** where more than one intentional crop type is grown together.

---

## Citation

If you use WeedOut in your research, please cite:

```bibtex
@inproceedings{kethineni2023weedout,
  title     = {WeedOut: An Autonomous Weed Sprayer in Smart Agriculture Framework
               using Semi-Supervised Non-CNN Annotation},
  author    = {Kethineni, Kiran Kumar and Mitra, Alakananda and
               Mohanty, Saraju P. and Kougianos, Elias},
  booktitle = {IFIP International Internet of Things Conference (IFIP-IoT)},
  year      = {2023}
}
```

---

## Authors

| Name | Affiliation | ORCID |
|---|---|---|
| **Kiran Kumar Kethineni** | Dept. of CS & Engineering, University of North Texas, USA | [0009-0004-6853-6749](https://orcid.org/0009-0004-6853-6749) |
| **Alakananda Mitra** | Nebraska Water Center, University of Nebraska-Lincoln, USA | [0000-0002-8796-4819](https://orcid.org/0000-0002-8796-4819) |
| **Saraju P. Mohanty** | Dept. of CS & Engineering, University of North Texas, USA | [0000-0003-2959-6541](https://orcid.org/0000-0003-2959-6541) |
| **Elias Kougianos** | Dept. of Electrical Engineering, University of North Texas, USA | [0000-0002-1616-7628](https://orcid.org/0000-0002-1616-7628) |

---

<div align="center">

**[Project Page](https://kirankkethineni.github.io/WeedOut/) | [View Notebook](WeedOut.ipynb)**

*Published at IFIP-IoT 2023 | University of North Texas*

</div>
