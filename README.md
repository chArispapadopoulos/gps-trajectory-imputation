# Imputation of Incomplete GPS Trajectories with Deep Learning Models Trained via Masking

Diploma thesis, School of Electrical and Computer Engineering, Aristotle University of Thessaloniki.
Author: Charalampos-Panagiotis Papadopoulos · Supervisor: Prof. Anastasios Delopoulos

GPS recordings from phones and wearables often stop for minutes or hours, because of signal loss or because the sensor is switched off to save battery. This project fills in those gaps. It compares two approaches:

- a **Transformer encoder** trained with masking (in the spirit of BERT) that regresses the missing coordinates directly, and
- a **modified version of the statistical method of Barnett & Onnela (2020)**, used as the baseline.

Both methods learn only from the history of a single user. No road network is needed at inference time.

## Method in short

**Data.** 170 days (May–December 2025) of GPS data from one user, sampled about once per minute. After preprocessing, 192,195 points remain. Each point gets a state flag (pause, flight, gap, break). The recording is then split into continuous segments, and stops and trips are detected.

**Input.** Windows of 48 points (about 48 minutes) with 13 features per point: position relative to home, displacement, time of day and day of week as sin/cos, time step, segment start/end, `is_masked` and `gap_frac`.

**Model.**
- Conv1d embedding (kernel 3) plus learned positional embedding.
- 3 Pre-LN Transformer encoder layers, d = 128, 4 heads, about 623k parameters.
- Regression head that outputs continuous coordinates.
- 3 rounds of iterative refinement: each round's prediction is fed back as input.

**Masking.**
- A contiguous span of L = 2…31 points is hidden, with p(L) ∝ 1/√L.
- A curriculum raises the maximum L from 3 to 31.
- Windows with movement are oversampled.
- A few **waypoints** inside the gap stay visible, chosen with trajectory simplification criteria (PED/SED). They help the model pick the right road instead of averaging between alternatives.

**Loss.**
- MIT: error on the hidden points.
- ORT: reconstruction of the visible points, as in SAITS.
- Optional step-length term.
- The rounds are weighted 0.3 / 0.3 / 1.

**Baseline.** Barnett & Onnela flight/pause sampling from the user's history with two changes:
1. a Student-t kernel (ν = 30) instead of Cauchy, so events near home no longer dominate;
2. a recursive bridging correction that keeps the shape of the simulated path.

**Evaluation.**
- 4-fold cross-validation over consecutive time periods, with a validation week inside each development set.
- Test gaps of L = 3, 6, 11, 20, 31 points in each window.
- Gaps are split into those with movement (≥ 500 m) and those without.
- Results are also broken down by *familiarity*: on how many days the user has passed through the area of the gap.
- 95% confidence intervals come from bootstrapping whole gaps.

## Main results

Main variant V9 (ORT, learning rate 10⁻³), on the 2,519 test gaps with movement, with waypoints:

| Metric | Barnett & Onnela (modified) | Transformer (V9) |
| --- | --- | --- |
| Median position error | 172 m | **105 m** |
| RMSE, all gaps | **468 m** | 617 m |
| RMSE, familiarity ≥ 10 days | 395 m | **297 m** |
| RMSE, unfamiliar gaps (76) | **1,138 m** | 3,032 m |
| Gaps where the model is better | | **68%** |

- The model usually finds the right path: the median distance from the true path is 27 m. Most of the remaining error is in *timing* along the path.
- Without waypoints the error roughly doubles.
- Familiarity drives the error. The baseline is safer in areas the user rarely visits, which is why its overall RMSE is lower. Without the two hardest days, the model's RMSE is 353 m vs 467 m for the baseline.

## Repository structure

```
thesis_EN.pdf                 full thesis in English
thesis.pdf                    full thesis in Greek (original)
Thesis_Presentation_EN.pptx   defense slides in English
Thesis presentation.pptx      defense slides in Greek (original)
preprocessing.ipynb           raw export -> cleaned dataset (final_dataset_thess9.csv)
train_eval_fold.ipynb         segments, features, windows, masking, training and test predictions for one fold
barnett_onnela.ipynb          modified Barnett & Onnela baseline on the same test gaps
analysis_all_folds.ipynb      tables and figures of the thesis from the outputs of all folds
```

## How to run

1. Install the dependencies:
   ```
   pip install numpy pandas scipy matplotlib torch pyproj shapely osmnx networkx
   ```
   `osmnx` is needed only for the optional route augmentation.
2. Run `preprocessing.ipynb` on your own GPS export to produce `final_dataset_thess9.csv`. Set `HOME_LAT` / `HOME_LON` to your own home location.
3. Run `train_eval_fold.ipynb` once per fold. Set `FOLD = 1…4` and use a separate folder per run. A GPU is recommended.
4. Run `barnett_onnela.ipynb`. It reads the test gaps saved by the model runs (`pred_fold{F}_*.csv`).
5. Run `analysis_all_folds.ipynb` to build the result tables and figures. The `RUNS` dictionary at the top maps each variant of the thesis to the output folder of its training runs; edit the paths to match yours.

## Data

The GPS data are **not included**, because they reveal the home location and daily routine of a real person. To reproduce the pipeline, use your own data with columns for timestamp, latitude and longitude. See the first cells of `preprocessing.ipynb` for the expected input format.

## References

- I. Barnett and J.-P. Onnela, "Inferring mobility measures from GPS traces with missing data," *Biostatistics*, 21(2):e98–e112, 2020.
- N. M. Gardazi et al., "BERT applications in natural language processing: a review," *Artificial Intelligence Review*, 58:166, 2025.

## Citation

```
@mastersthesis{papadopoulos_gps_imputation,
  author = {Charalampos-Panagiotis Papadopoulos},
  title  = {Imputation of Incomplete GPS Trajectories with Deep Learning Models Trained via Masking},
  school = {Aristotle University of Thessaloniki, School of Electrical and Computer Engineering},
  year   = {2026},
  note   = {Diploma thesis. Supervisor: Anastasios Delopoulos}
}
```