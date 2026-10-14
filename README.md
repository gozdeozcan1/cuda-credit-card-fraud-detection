# Credit Card Fraud Detection with C++, OpenMP, and CUDA

C++ implementations of Logistic Regression, linear Support Vector Machine, and Random Forest for serial CPU, OpenMP, and CUDA execution, prepared from a Ph.D. course project on credit card fraud detection.

All three model families from the accompanying report are included. Each model/backend combination has its own executable, for nine build targets in total.

## What the project demonstrates

- Logistic Regression with L2 regularization, a linear hinge-loss SVM, and histogram-based Random Forest training in C++.
- CPU parallel gradient accumulation with OpenMP.
- CUDA sample-level gradient calculation and shared-memory block reductions.
- GPU prediction and CPU/GPU training-time comparison.
- CSV validation, reproducible train/test splitting, and training-only feature standardization.
- Accuracy, precision, recall, F1, and confusion matrix counts.

## Dataset

Source: [Credit Card Fraud Detection Dataset 2023 — Nidula Elgiriyewithana, Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023).

Download the dataset yourself and place `creditcard_2023.csv` under a local `data/` folder. No input transactions or derived datasets are distributed in this repository. The expected feature columns are `V1` through `V28` and `Amount`; `Class` is the binary label. An optional `id` column is excluded from the model. The original report describes 568,630 rows and equal class counts; the runner prints actual accepted counts for your local file.

This uses the 2023 dataset, not the separate ULB dataset containing a `Time` column.

## Data preparation

The uploaded model implementations already call training-set standardization. Their original `Common.h` implementation and program entry point were not supplied. This repository adds a documented common implementation and runner; it does not claim to reproduce the missing original data loader exactly.

| Step | Behavior in this repository |
|---|---|
| Schema validation | Requires `V1`–`V28`, `Amount`, and `Class`; rejects missing/duplicate required columns and unexpected columns. |
| Record validation | Skips malformed rows, missing or non-finite feature values, negative amounts, and labels outside 0/1; prints rejected counts. |
| Feature selection | Excludes `id` and the target; uses 29 numerical features in a fixed order. |
| Train/test split | Seed 42; shuffles within each class and reserves approximately 20% per class for testing. |
| Standardization | Computes training means and population standard deviations; applies the same values to the test set. Constant columns use a divisor of 1. |
| CUDA layout | Converts standardized features to contiguous row-major float buffers and labels to byte buffers. |

No missing-value imputation, duplicate removal, statistical outlier filtering, SMOTE, oversampling, or dataset expansion is performed by this runner. The report describes historical data cleaning and expansion, but the associated scripts were not supplied. See [preparation notes](docs/PREPARATION_NOTES.md) for the distinction.

## Model and execution paths

### Logistic Regression

The three LR implementations use 200 epochs, learning rate 0.01, L2 coefficient 0.01, decay 0, zero initialization, and a probability threshold of 0.5. Their gradients correspond to binary cross-entropy plus L2 regularization.

Serial execution accumulates sample gradients sequentially. OpenMP combines per-thread gradients in a critical section. CUDA computes shared-memory block reductions, copies partial gradients to the CPU, and updates weights on the CPU. Final prediction is performed on the GPU.

### Linear SVM

The supplied serial/OpenMP function names contain `SGD`, but inspection shows gradients are accumulated over the full dataset before each epoch update. They use unnormalized summed gradients, C=1, 200 epochs, learning rate 0.01, and decay 0.01. CPU uses doubles while CUDA uses floats and parallel reduction ordering. The source snapshots are not a matched isolated speed benchmark.

Auxiliary `slowbuf`/`slow_barrier` work in the supplied CPU SVM implementations was removed because it does not contribute to model learning. Historical report runtimes cannot be reproduced by assuming that this cleaned version is the original benchmark source. Model-specific settings remain in the source; see preparation notes.

### Random Forest

Serial/CUDA RF use 50 trees, while the supplied OpenMP version uses 35 trees; all use maximum depth 10, bootstrap sampling, random feature subsets, and histogram-based candidate splits. OpenMP parallelizes tree building and prediction. The CUDA implementation retains host tree construction while accelerating split histograms and prediction on the GPU. Random seeds, precision, and execution order differ between backends; exact tree equivalence is not claimed.

All model runners standardize using training statistics. CUDA implementations are hybrid CPU/GPU algorithms. The original report's statement that LR weights are updated directly on the GPU does not describe this uploaded source snapshot.

## Build

Requirements: a C++17 compiler and CMake 3.24+. OpenMP is enabled by default. CUDA is optional and requires an NVIDIA GPU, CUDA Toolkit, and a compatible host compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

To enable CUDA:

```sh
cmake -S . -B build -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

To build serial only, add `-DBUILD_OPENMP=OFF`. The model/backend implementations use separate executables to avoid duplicate helper symbols and allow independent timing.

### Run on Windows / Visual Studio

Run from the repository root after downloading the CSV:

```powershell
.\build\Release\fraud_lr_serial.exe data\creditcard_2023.csv
.\build\Release\fraud_lr_openmp.exe data\creditcard_2023.csv
.\build\Release\fraud_lr_cuda.exe data\creditcard_2023.csv
```

Replace `lr` with `svm` or `rf` in executable names to run the other models, for example `fraud_svm_serial.exe` or `fraud_rf_cuda.exe`. All nine names follow `fraud_<lr|svm|rf>_<serial|openmp|cuda>`.

For a quicker check, add a maximum accepted-row count:

```powershell
.\build\Release\fraud_lr_serial.exe data\creditcard_2023.csv 20000
```

The limit selects an input prefix, not a stratified sample of the whole file. If that prefix contains fewer than two records of either class, use a larger limit or omit it. Zero means all rows. Use the same input and limit for every executable. Loading all records needs substantial host memory because the source models copy data and CUDA builds additional float buffers.

### Run on Linux / single-configuration builds

```sh
./build/fraud_lr_serial data/creditcard_2023.csv
OMP_NUM_THREADS=4 ./build/fraud_lr_openmp data/creditcard_2023.csv
./build/fraud_lr_cuda data/creditcard_2023.csv
```

Without CMake, the serial and OpenMP versions can also be built directly:

```sh
g++ -std=c++17 -O2 src/main.cpp src/Common.cpp src/SerialLR.cpp -o fraud_lr_serial
g++ -std=c++17 -O2 -fopenmp -DFRAUD_OPENMP src/main.cpp src/Common.cpp src/LROpenMP.cpp -o fraud_lr_openmp
```

## Historical report results

These are results for all three models transcribed from page 5 of the supplied report, not measurements reproduced during repository preparation.

| Dataset | Model | Serial accuracy | CUDA accuracy | Serial execution | CUDA execution | Reported speedup |
|---|---|---:|---:|---:|---:|---:|
| Original | LR | 0.926068 | 0.926068 | 49.986 s | 1.594 s | 31.35× |
| Original | SVM | 0.996817 | 0.996817 | 20.016 s | 1.522 s | 13.15× |
| Original | RF | 0.999631 | 0.999666 | 265.963 s | 115.320 s | 2.31× |
| Expanded / resampled | LR | 0.927035 | 0.927049 | 519.38 s | 4.999 s | 103.88× |
| Expanded / resampled | SVM | 0.996863 | 0.996867 | 109.35 s | 4.372 s | 25.01× |
| Expanded / resampled | RF | 0.999813 | 0.999789 | 1767.03 s | 868.894 s | 2.03× |

The exact historical split, cleaning implementation, expansion script, hardware configuration, and raw logs are unavailable. The added loader may produce different results. No OpenMP historical measurements are supplied in the report tables. See [benchmark notes](benchmarks/README.md).

## Validation

All six serial and OpenMP executables compiled with GCC in C++17 mode with warnings enabled. All six passed a temporary synthetic-data smoke check, record rejection checks, and missing-file/schema/invalid-limit checks. Standardization was checked independently. No input data or test data are included in the package.

CUDA compilation/execution and CMake configuration have not been verified in this environment. Full Kaggle training was not run. Successful small-data checks do not validate the historical accuracy or speedup claims.

## Repository contents

- `src/`: comment-free implementations of all three models, headers, common preprocessing, and CLI runner.
- `CMakeLists.txt`: serial/OpenMP targets and an optional CUDA target.
- `benchmarks/reported_results.csv`: historical measurements for all three models.
- `benchmarks/README.md`: timing boundaries and reproduction guidance.
- `docs/PREPARATION_NOTES.md`: selected scope, added files, changes, and evidence limits.
- `docs/original_report.pdf` and `docs/original_presentation.pptx`: unchanged documents covering the broader three-model study.

## Attribution

Prepared for Gözde Özcan's portfolio. The original report lists Gözde Özcan and Esin Onbaşıoğlu as authors. Model references are preserved in that report. No publication venue or peer-reviewed publication status is claimed for this course project.

No open-source license is assigned. Dataset access and usage follow the source dataset's terms.
