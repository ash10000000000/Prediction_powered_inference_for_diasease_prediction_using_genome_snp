Prediction-Powered Inference (PPI) for Genome SNP Disease Probability Estimation

This repository implements Prediction-Powered Inference (PPI) to estimate disease probability from synthetic Single Nucleotide Polymorphism (SNP) data. The project demonstrates how PPI can combine machine learning predictions from large unlabeled datasets with limited labeled data to produce narrower confidence intervals (CIs) while maintaining statistical validity.

Project Overview

Goal: Compare PPI, classical, and imputation approaches in estimating disease probability from genotype data (0, 1, 2).

Model: Logistic regression with calibration for better probability estimation.

Simulation:

100,000 total samples (genotypes) generated synthetically.

10,000 samples used for training, rest unlabeled.

50 labeled samples per trial across 100 trials.

Metrics:

CI width → Precision of estimate.

Coverage → Reliability (how often the true value lies within CI).

Features & Enhancements

Sigmoid calibration using CalibratedClassifierCV for improved probability estimates.
Trial skipping when labeled sets are too imbalanced (<2 positives/negatives).
Fixed seeds for reproducibility across trials.
Coverage calculation to assess reliability.
Visualization outputs:

ci_width_bar.png → CI width comparison.

ci_width_boxplot.png → Distribution of CI widths across trials.

coverage_bar.png → Coverage comparison between methods.

Results
Method	CI Width	Coverage
PPI	0.235	0.930
Classical	0.277	0.940
Imputation	0.003	N/A (unrealistic)

CI width reduction: ~15% (PPI vs. Classical)

Coverage: PPI coverage close to nominal 95% target.

Key Insight: PPI provides more precise estimates without sacrificing coverage.
