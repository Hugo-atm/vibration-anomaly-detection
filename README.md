# Vibration-Spectrum Analysis and Anomaly Detection

**Exploratory machine-learning and data-analysis project · Python / Jupyter**

This repository documents experiments aimed at identifying unusual patterns in rotating-equipment vibration measurements. It explores spectral feature preparation, conventional outlier-detection algorithms, and neural-network approaches. The code originated as exploratory, notebook-driven work: **it is not a production predictive-maintenance system, and it does not demonstrate a validated equipment-failure predictor.**

## Background

Vibration monitoring can help identify changes in rotating machinery. Measurements are often noisy and differ across equipment and operating conditions. Instead of using a single scalar measurement, the scripts investigate **frequency-band peaks**: a frequency and its corresponding vibration amplitude for each measured spectral band.

In the accompanying vibration-sensor design documents, a portable measurement concept uses **Banner/Turck QM30VT2** vibration/temperature sensors with a proposed path to server-side collection and analysis. The design brief calls for peak-component RMS amplitude and peak-component frequency within configurable spectral bands, together with vibration and temperature measurements. The **sensor hardware, Wi-Fi transfer, Modbus setup, and database ingestion are not implemented in these Python scripts**; they are background context, not a claim of a functioning end-to-end system.

The manufacturer's *VT2 Vibration Spectral Band Configuration* technical note explains that a band's **peak-component RMS amplitude** is the RMS value of its largest spectral component and the **peak-component frequency** is the frequency of that component. These are distinct from the *total* RMS amplitude over a band. The documentation also distinguishes fixed manual bands from speed-dependent configurable bands.

## Research objectives

1. Explore and visualize frequency/amplitude observations from vibration-spectrum CSV files.
2. Clean and reshape multiple recordings into machine-learning-friendly tables.
3. Compare statistical and machine-learning ways of flagging unusual observations.
4. Investigate whether autoencoders and GAN-based experiments can support anomaly analysis.
5. Identify challenges posed by limited data, variable distributions, and the absence of verified fault labels.

**Important:** An outlier score is *not* proof of a mechanical fault. No verified diagnostic sensitivity, specificity, predictive-maintenance savings, or operational deployment outcome is claimed here.

## Observed data structures

The scripts use several intermediate CSV formats rather than one standardized dataset.

| Fields / files | Meaning in these experiments |
| --- | --- |
| `f1`–`f8` | Frequency values from multiple bands or peaks |
| `v1`–`v8` | Corresponding vibration amplitudes / velocities |
| `source_file` | Identifier retained when combining individual recordings |
| `max_f`, `max_v` | Frequency and amplitude associated with a row's largest recorded velocity |
| `f`, `v` | Reshaped or grouped frequency/amplitude pairs in anomaly experiments |
| `compile_clear.csv`, `data_round.csv`, `avgdatav.csv`, `compile_all.csv` | Intermediate or prepared inputs referenced by separate notebook stages |

The **exact mapping of script column indices to configured hardware spectral bands is not fully documented** in the code and should not be assumed from the names alone. The sensor design brief lists illustrative band boundaries, while the notebooks also explore folder groupings such as `10-20` and `20-30`; these should not be treated as one verified, identical configuration.

## What the scripts do

### 1. Data exploration and preparation — `scripts/stat.py`

- Iterates across folders containing vibration-spectrum CSV recordings.
- Generates frequency-versus-velocity scatter and stem plots.
- Combines band measurements and preserves source filenames for traceability.
- Calculates peak velocity and its associated frequency.
- Builds intermediate CSV tables and grouped summaries.
- Explores k-nearest-neighbor classification, curve fitting, and further anomaly plots.

This is a long exploratory script containing repeated code blocks and commented-out alternatives, **not a clean, ordered command-line data pipeline**. Some sections assume prior variables or CSV outputs.

### 2. Cleaning and multi-output regression — `scripts/cleaning_data_2.py`

- Loads `data_round.csv`.
- Uses the interquartile range (IQR) to **cap** numeric outliers at the 1.5 × IQR bounds, rather than deleting outlier rows.
- Uses frequencies `f1`–`f8` as inputs and velocities `v1`–`v8` as prediction targets.
- Tries a `MultiOutputRegressor` wrapping `RandomForestRegressor`, with `GridSearchCV` for tree depth.
- Reports regression metrics and plots feature-versus-predicted-value relationships.

**This portion investigates regression**, not confirmed fault classification. It depends on the third-party `jcopml` package as written.

### 3. Classical unsupervised anomaly detection — `scripts/final_isof_lof_ee.py`

Uses `avgdatav.csv`, groups observations by discrete `f` values, and explores:

- **Isolation Forest**, with a specified contamination setting (0.1 in the main experiment).
- **Local Outlier Factor (LOF)**, with neighbor counts adjusted to the available sample size.
- **Elliptic Envelope**, a robust covariance-based outlier detector.

The code produces per-frequency plots and tables of selected anomalous source files. Some sections attempt to compare outliers identified by more than one detector. The contamination setting is a **model assumption** about expected outlier proportion, not a measured fault rate. Some cells reference previously defined variables, and not every statement will execute independently.

### 4. Neural-network experiments — `scripts/anomali.py`

- Implements a dense **autoencoder** for reconstructing a vibration feature and identifies large reconstruction errors using a percentile cutoff.
- Defines **generator** and **discriminator** networks using TensorFlow/Keras, and experiments with GAN training on scaled vibration data.
- Investigates generated samples, gradient behavior, and anomalous records.

These are **experimental architectures**, not validated deep-learning benchmarks. In particular, the GAN-related anomaly-reconstruction logic requires methodological review; generating samples from random latent inputs is not automatically a valid per-observation reconstruction.

## Repository layout

```text
vibration-anomaly-detection/
├── README.md
├── .gitignore
└── scripts/
    ├── stat.py
    ├── cleaning_data_2.py
    ├── final_isof_lof_ee.py
    └── anomali.py
```

[Browse the scripts](scripts/). Original Jupyter notebooks from the wider research collection are not currently part of this GitHub repository.

## Environment and execution

**Observed libraries:** Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, TensorFlow/Keras, and `jcopml` in the regression script. Some scripts were exported from Google Colab and contain notebook-specific syntax such as `!pip`, bare `pip install`, and `display(...)`.

To prepare a basic exploratory environment:

```bash
python -m venv .venv
# Activate your virtual environment
python -m pip install numpy pandas scipy matplotlib seaborn scikit-learn tensorflow notebook
# The multi-output regression script additionally references jcopml.
```

**Do not assume that running `python scripts/stat.py` will work unchanged.** Before executing an individual section, restore the necessary input CSV files, update its local working directories, correct notebook-only statements where applicable, and inspect variable dependencies. No fresh-environment integration test or package-version pinning has been completed.

## Study limitations and research integrity

- **No published ground-truth fault labels.** The scripts flag statistical outliers; they do not demonstrate reliable mechanical fault diagnoses.
- **No comparable benchmark results.** Different sections use different data shapes and thresholds; accuracy rankings or superior model performance cannot be inferred.
- **Small-sample concerns.** Local models may be unstable when few observations exist per frequency group.
- **Potential data leakage.** Data preparation, scaling, and splits need review before treating regression or anomaly metrics as out-of-sample evidence.
- **Unclear GAN evaluation.** The generative experiments should be treated as learning exercises until they have a well-defined evaluation protocol.
- **Data and authorship review.** Input vibration measurements are not published here, and some code may have been developed collaboratively or adapted from other sources. Attribution and reuse permissions should be checked before presenting each module as solely original work.
- **Employer information.** The associated hardware design document is contextual reference material, not included in this repository; publishing operating data or confidential designs requires authorization.

## Potential next steps

1. Create a documented, synthetic sample dataset with explicit spectral-band definitions and units.
2. Split exploration, preprocessing, and modeling into independently executable modules.
3. Use equipment-aware time splits and held-out machines to assess generalization.
4. Compare anomaly detectors with matched contamination and consistent evaluation.
5. Add true maintenance/fault labels where available and authorized; investigate calibration and uncertainty of anomaly scores.

## Related technical references

- Internal project design brief: *Remote PM Vibration Sensor* (project context; not redistributed here).
- Banner Engineering: *VT2 Vibration Spectral Band Configuration*, technical note **b_4510565 Rev. D** (configuration and interpretation of peak-component measurements).

---
**Portfolio purpose:** To document hands-on experimentation in signal analysis, noisy-data processing, unsupervised learning, and deep-learning approaches. The scripts record exploratory work, not an independently validated product.