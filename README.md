# GPS Trajectory Imputation with a Masked Transformer

Diploma thesis, School of Electrical and Computer Engineering, Aristotle University of Thessaloniki (2026).
Supervisor: Prof. Anastasios Delopoulos.

GPS traces are often interrupted for minutes or hours, because the sensor is switched off or loses signal. This thesis fills such gaps with a Transformer encoder trained with masking, and compares it with the statistical method of Barnett \& Onnela (2020) on exactly the same gaps.

The full thesis is in [`thesis.pdf`](thesis.pdf) (in Greek) and the defence slides in `Thesis presentation.pptx`. The markdown cells of the notebooks are also in Greek.

## Overview

**Data.** GPS traces of a single user in Thessaloniki, 170 days in total, exported as recording sessions and turned into one table of points with time offsets `dt` and displacements `dx, dy`. Coordinates are projected to EGSA'87 (EPSG:2100), so distances are in metres.

**Model.**

* Window of 48 points. A gap of 2–31 points is hidden inside it. Hidden positions are set to a sentinel value and the model predicts their coordinates (regression, no classifier head).
* Conv1d input layer, learned positional embeddings and a 3-layer pre-LN Transformer encoder (`d\_model` 128, 4 heads, dropout 0.3).
* 13 input features per point: position relative to home, displacement, time of day and day of week (sin/cos), time step, segment boundaries, mask flag and relative position inside the gap.
* 0–3 visible waypoints can be placed inside a gap to disambiguate which route was taken.
* 3 refinement rounds: the predictions of one round are fed back as input to the next.
* Loss: error on the hidden points (MIT) plus a weighted error on the visible ones (ORT).
* Training uses a curriculum on the gap length (3 → 31) and a sampler that favours windows on trips and in movement.

**Baseline.** Own implementation of the Barnett \& Onnela flight/pause simulation (kernel with a Student-t weight, mean of 100 simulations) with a modified bridging step.

**Evaluation.**

* 4-fold cross-validation over consecutive days. The validation set is the last full week of each non-test fold.
* Gap lengths 3, 6, 11, 20 and 31 points, evaluated with and without waypoints and separately for gaps with and without movement.
* Metrics: RMSE with a bootstrap 95% confidence interval, median and p90 error, path error and familiarity of the area.
* Twelve model variants compared (loss terms, route augmentation with OSM map matching, learning rate, model size, dropout). The main variant is "Π9".

Results are in Chapter 4 of the thesis. `analysis\_all\_folds.ipynb` reproduces its Tables 4.1–4.12 and the training-curve figure.

## Files

```
preprocessing.ipynb        raw export -> cleaned dataset (final\_dataset\_thess9.csv)
train\_eval\_fold.ipynb      segments, trips, windows, masking, training and evaluation of one CV fold
barnett\_onnela.ipynb       Barnett \& Onnela baseline on the same gaps
analysis\_all\_folds.ipynb   tables and figures over the 4 folds
thesis.pdf                 full thesis (Greek)
Thesis presentation.pptx   defence slides
```

## How to run

The data and the run outputs are not in this repository, so the notebooks show the pipeline rather than run out of the box.

1. `preprocessing.ipynb`: set `HOME\_LAT` and `HOME\_LON` to your own home location. Input is `export\_gps\_v1.csv` (sessions with latitude, longitude and elapsed time). Output is `final\_dataset\_thess9.csv`.
2. `train\_eval\_fold.ipynb`: run once per fold with `FOLD` set to 1–4, each in its own folder with the dataset. A GPU is recommended. The training runs were done without a fixed seed, so a new run gives slightly different numbers.
3. `barnett\_onnela.ipynb`: reads the gaps from the prediction files of step 2, so the two methods are compared on identical gaps.
4. `analysis\_all\_folds.ipynb`: aggregates the outputs of steps 2 and 3. The expected folder layout is set at the top of the notebook (`ROOT`, `RUNS`, `BARNETT`).

## Requirements

Python with `numpy`, `pandas`, `scipy`, `pyproj`, `matplotlib` and `torch`. Only the optional route augmentation needs `osmnx`, `networkx` and `shapely`. Training ran on Google Colab and on the AUTH Aristotelis HPC cluster (SLURM, one GPU per fold).

## Data

The GPS dataset is personal data and is not included in this repository.

## Reference

Barnett, I., \& Onnela, J.-P. (2020). Inferring mobility measures from GPS traces with missing data. *Biostatistics*, 21(2), e98–e112.

