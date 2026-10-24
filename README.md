# Vision Transformers for Satellite Image Classification: Zero-Shot, Linear Probe and Fine-Tuning, with a Robustness Study

**Author:** [Muskan Khurana] | **Course / Programme:** [MAAI901] | 
## Overview

This project applies pre-trained transformer models to a four-class satellite land-cover dataset
(`cloudy`, `desert`, `green_area`, `water`) and asks:

> How well do zero-shot, frozen-feature (linear probe) and fine-tuned vision transformers classify satellite images,
> and do their predictions rely on real scene content or on dataset shortcuts such as image resolution and colour?

| Method | Model |
|---|---|
| Zero-shot classification | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) |
| Linear probe on frozen embeddings | CLIP ViT-B/32 and DINOv2-base (`facebook/dinov2-base`) |
| Fine-tuning | Supervised ViT (`google/vit-base-patch16-224`) |
| Stress tests / retrieval / statistics | all of the above |

## Dataset

* Source: **[https://www.kaggle.com/code/sujithmandala/satellite-image-classification-cnn/input]**
* 5,631 images: cloudy 1,500 · desert 1,131 · green_area 1,500 · water 1,500
* **Dataset artefact found:** cloudy and desert images are 256x256, while green_area and water images are 64x64, so
  image size alone predicts the class pair (a size-only rule scores about 53% vs 25% chance). A colour-only baseline
  (mean R, G, B) reaches about 82%. The robustness notebook tests whether the models exploit this.
* The data are **not included** in this repository; see `data/README.md`.
* Split: duplicate-aware (exact and near-duplicate images are grouped), stratified 70/15/15 → 3,942 train / 846 validation /
  843 test images, seed 42. The split is stored in `data/split.csv`.

## Notebooks (run in this order)

| # | Notebook | Colab | Inputs | Produces | Approx. time (T4) |
|---|---|---|---|---|---|
| 1 | `01_setup_explore_zeroshot_clip.ipynb`: audit, duplicate check, split, zero-shot CLIP | [Open](https://colab.research.google.com/github/muskankh/Satellite-vit-project/blob/main/notebooks/01_setup_explore_zeroshot_clip.ipynb) | `data-2.zip` | `split.csv`, CLIP embeddings, results JSON | ~5 min |
| 2 | `02_linear_probe_clip_dinov2.ipynb`: linear probes, learning curve | [Open](https://colab.research.google.com/github/muskankh/Satellite-vit-project/blob/main/notebooks/02_linear_probe_clip_dinov2.ipynb) | `split.csv`, `data-2.zip` | embeddings, results JSON | ~10 min |
| 3 | `03_finetune_vit.ipynb`: fine-tune ViT, LR search, 3 seeds | [Open](https://colab.research.google.com/github/muskankh/Satellite-vit-project/blob/main/notebooks/03_finetune_vit.ipynb) | `split.csv`, `data-2.zip` | saved model, predictions, results JSON | ~20-30 min |
| 4 | `04_robustness_retrieval_stats.ipynb`: robustness, statistics, retrieval, UMAP | [Open](https://colab.research.google.com/github/muskankh/Satellite-vit-project/blob/main/notebooks/04_robustness_retrieval_stats.ipynb) | `split.csv`, `data-2.zip`, model from Notebook 3 | figures, results JSON | ~10-20 min |

## How to reproduce

1. Open a notebook in Google Colab (button above) and set **Runtime → Change runtime type → T4 GPU**.
2. Drag `data-2.zip` and `data/split.csv` into the Colab **Files** panel (or keep them in `My Drive/satellite_project/`).
3. Each notebook has a `QUICK_TEST = True` option in its config cell: a small dry run to check that everything works.
   Set it back to `False` for the real run (quick-test numbers are meaningless).
4. Run the notebooks in order. Notebook 4 needs the model saved by Notebook 3 (`vit_best_model/`).
5. All random seeds are fixed (`SEED = 42`; fine-tuning uses seeds 42, 43, 44). Tiny numeric differences can occur
   between GPU types and library versions.

## Results summary

Test set = 843 images. 95% confidence intervals come from bootstrap resampling of the test set.

| Method | Test accuracy | Macro-F1 |
|---|---|---|
| Chance | 25.0% | n/a |
| Image-size-only rule (reference) | 53.3% | n/a |
| Colour-only logistic regression (reference) | 82.4% | n/a |
| Zero-shot CLIP ViT-B/32 (prompt: rich class names, "a satellite image taken from space showing …") | 74.6% (CI 71.5-77.6) | 74.8% |
| Linear probe: CLIP ViT-B/32 (C = 0.01) | 99.88% (CI 99.5-100.0) | 99.89% |
| Linear probe: DINOv2-base (C = 0.01) | 99.64% (CI 99.2-100.0) | 99.67% |
| Fine-tuned ViT (mean ± std over 3 seeds; lr 1e-5, 4 epochs) | 99.96 ± 0.07% | 99.96 ± 0.06% |

The three trained methods are statistically indistinguishable on this test set (0-3 errors in 843 images; paired bootstrap and
McNemar, Bonferroni-corrected). Only zero-shot CLIP differs (about 25 points lower, p < 1e-60). Fine-tuned ViT: seed 42 = 99.88%
(CI 99.6-100.0), 3-seed ensemble = 100%.

Robustness: test accuracy (%) when the test images are altered (models not retrained; see `results/results_notebook4.json`
and `results/robustness_drops.csv`). All changes are significant (paired 95% bootstrap CI excludes zero) except zero-shot with
same-resolution and the DINOv2 probe when darkened.

| Model | Original | Same resolution (all 64x64) | Greyscale | Hue shift | Darker | Brighter |
|---|---|---|---|---|---|---|
| Zero-shot CLIP | 74.6 | 73.5 | 55.5 | 40.5 | 65.2 | 61.8 |
| CLIP probe | 99.9 | 87.8 | 77.0 | 88.6 | 90.6 | 89.3 |
| DINOv2 probe | 99.6 | 98.5 | 88.5 | 98.3 | 99.4 | 99.1 |
| Fine-tuned ViT (seed 42) | 99.9 | 98.2 | 89.9 | 98.2 | 75.8 | 87.9 |

Key findings: the CLIP probe loses 12.1 points when the resolution cue is removed (cross-group error 0% to 12.0%), consistent
with partial use of resolution/sharpness, while DINOv2 and the fine-tuned ViT lose under 2 points. The fine-tuned ViT is the most
fragile to darkening (-24.1 points). Greyscale costs every model 10-23 points. Retrieval: image-to-image precision@1 is 99.2%
(CLIP) and 99.5% (DINOv2); CLIP text-to-image retrieval works for clouds, desert and forest but fails for water queries.
Compute for all fine-tuning runs: about 0.27 GPU-hours on a Tesla T4.

Reference baselines (size-only, colour-only) were computed on a separate random split in Notebook 1 and are shown for context.

## Repository structure

```
notebooks/   four notebooks (run in order)
data/        split.csv + data description (images are not included)
results/     JSON/CSV result files from each notebook
figures/     figures used in the report
report/      technical report (PDF and Word)
slides/      presentation (PowerPoint)
requirements.txt
```

## Trained model

The fine-tuned model (about 330 MB) is too large for GitHub.
Download: **[https://github.com/muskankh/Satellite-vit-project/releases/tag/v1.0-submission]**. Place the folder `vit_best_model/` next to the notebooks (or in
`My Drive/satellite_project/outputs_nb3/`) before running Notebook 4.

## Limitations

* One dataset with four coarse classes and a single train/validation/test split.
* The learning-rate search uses one seed and a small grid.
* Robustness perturbations are synthetic; a drop under a perturbation is evidence of sensitivity, not proof of the cause.
* Image size and class are confounded in the data, so their effects cannot be fully separated without new data.

## Ethics and responsible use

Satellite-image classifiers can support environmental monitoring but are dual-use (surveillance). The geographic coverage of
the dataset is unknown, so results may not transfer to other regions, sensors or seasons. See the report for discussion.


## References

Dosovitskiy et al. (2021) ViT, ICLR · Radford et al. (2021) CLIP, ICML · Oquab et al. (2024) DINOv2, TMLR · Geirhos et al. (2020)
Shortcut learning, Nature Machine Intelligence · Loshchilov and Hutter (2019) AdamW, ICLR · Wolf et al. (2020) Transformers, EMNLP
Demos · Pedregosa et al. (2011) scikit-learn, JMLR · McInnes et al. (2018) UMAP, arXiv:1802.03426 · McNemar (1947) Psychometrika.
Full list in the report.
