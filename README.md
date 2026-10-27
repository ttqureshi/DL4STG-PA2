# DL4STG-PA2

LUMS AI651, Fall 2026. Assignment 2: Deep Learning for Space, Time and Graphs.

**Student:** Muhammad Tayyab Tahir Qureshi  
**Roll number:** 25280024  
**Report:** [`report.pdf`](report.pdf), built from [`report.tex`](report.tex)

The report is the submission. This repository holds the code that produced it, the trained weights, the tables, the figures, and the GIFs cited in the captions.

## What the two questions are

Question 1 is a design, not a trained model. It proposes a short-series attention forecaster for water level at four river stations, S1–S4. The write-up is in the report. Nothing in `Question1/` is submitted; that folder is the optional companion notebook.

Question 2 asks whether a ConvLSTM trained on Moving MNIST has learned how digits move, or only what the training videos usually look like. The tests are a horizontal mirror, a 90° anticlockwise rotation, and playback at 2× and 3×. The final model is Model D trained with both speed and symmetry augmentation, then wrapped so that it averages all eight symmetries of the square.

## Data

`torchvision.datasets.MovingMNIST` with `split=None`. That returns all 10,000 videos of 20 frames at 64×64. The dataset’s own train/test split is not used.

| Split | Indices | Role |
| --- | --- | --- |
| Train | 0–7,999 | One random legal window per video per epoch |
| Validation | 8,000–8,999 | Not used to pick a reported checkpoint |
| Test | 9,000–9,999 | Every reported score and prediction figure |

Frames in the report are numbered 1–20. Tensor index `f - 1` is write-up frame `f`. Pixels are float32 in `[0, 1]`.

## Models

All learned models emit five frames from five input frames, in a nine-step rollout with no teacher forcing. Steps 1–5 read real frames. Steps 6–9 read the network’s own previous prediction.

| Model | Architecture | Training |
| --- | --- | --- |
| A | Two ConvLSTM layers, 64 channels, 3×3 | 1× only, no augmentation |
| B | Same as A | 1×, one random symmetry from the group of the square on each window |
| C | Same as A | Speed drawn from {1×, 2×, 3×}, no symmetry |
| D | Same as A with dilation 2 on both layers | Same speed schedule as C |
| D+B | Same as D | Speed schedule of C and the symmetry draw of B |

A, B, C, D, and D+B each have 480,784 trainable parameters. Model D’s one-step reach is 16 original pixels; Model A’s is 8. The cap for D was 721,176. Seeds are 0 and 1. Tables report the mean, then the two seeds. Blank and Copy-last are deterministic.

The symmetrised forecaster needs no extra training:

```text
f_bar(X) = (1/8) sum over g in G of  g^{-1}( f(g(X)) )
```

The final model is that wrapper around D+B. Its mirror and rotation consistency gaps are floating-point zeros.

Training used Adam at `1e-3`, batch 32, 30 epochs, and the final-epoch weights. Runs were on a Kaggle Tesla T4. Q2.5 (`10cb5f4`) was pushed before the mirror, rotation, and speed tests. Q2.11 (`ede80b4`) was pushed before training B, C, and D.

## Layout

```text
src/dl4stg/          transforms, windows, ConvLSTM, models A and D, symmetriser
scripts/             Table 1, Figure 1, training, and evaluation entry points
kaggle/              private kernel metadata used to run those scripts on Kaggle
weights/             final-epoch state dicts for A, B, C, D, and D+B, seeds 0 and 1
results/             table1.json through table6_final.json, plus per-epoch logs
figures/             Figures 1–6; GIFs for Figures 3 and 6 are in figures/gifs/
drafts/              answer drafts, including the frozen Q2.5 and Q2.11 texts
report.tex           the report
Question1/           optional, ungraded companion notebook
```

Raw Moving MNIST downloads and intermediate checkpoints are gitignored (`data/`, `checkpoints/`).

## Reproducing a check

The geometry checks do not need the dataset:

```text
python scripts/table1.py
```

They must report maximum absolute difference 0 for the mirror and rotation identities, bright pixels `(7, 10, 43)` and `(7, 43, 10)`, and the 1×, 2×, and 3× windows in Table 1.

Figure 1 and the training scripts download Moving MNIST on first use. Local CPU PyTorch in this environment does not import (`WinError 1114` on `c10.dll`). The reported numbers were produced on Kaggle with the kernels under `kaggle/`. A working CUDA or CPU build can run the same scripts:

```text
pip install -r requirements.txt
python scripts/figure1.py
python scripts/train_a.py
```

`train_b.py`, `train_c.py`, `train_d.py`, and `train_db.py` are the later models. Evaluation entry points are `scripts/eval_sym_bd.py` and `scripts/eval_sym_final.py`. Do not use the test split to choose a design. The reported weights are the last epoch, not the best validation epoch.
