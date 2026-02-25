# Exoplanet Candidate Classification Using Ensemble Machine Learning on Large-Scale Public Data
## Overview

A rigorous comparative study of probabilistic classification models and ensemble methods applied to large, noisy observational data.

This project develops a reproducible end-to-end machine learning pipeline to classify exoplanet candidates using publicly available data from NASA's Transiting Exoplanet Survey Satellite (TESS) mission.

The objective was to evaluate and compare heterogeneous classification models and assess whether a soft-voting ensemble model improves probabilistic discrimination performance in a large, noisy observational dataset.

## Problem Statement

The TESS candidate catalogue contains large volumes of observational data used to identify potential exoplanets. Candidate classifications are derived from noisy photometric measurements and may include false positives due to astrophysical or instrumental effects.

The challenge is to develop a robust probabilistic classification framework capable of distinguishing confirmed planetary signals from non-planetary detections while maintaining reliable precision-recall trade-offs.

This project evaluates multiple machine learning approaches and investigates whether a soft-voting ensemble can improve classification stability and predictive performance across heterogeneous model families.

## Project Overview

Pipeline includes:
- Data preprocessing and cleaning
- Feature engineering and transformation
- Train/Test separation to prevent data leakage
- Model comparison across multiple algorithm families
- Precision-recall based evaluation
- Soft-voting ensemble implementation
- Threshold analysis and performance evaluation

## Models Evaluated

The following classifiers were implemented and tuned:
- Support Vector Classifier (SVC)
- Random Forest (RF)
- Gradient Boosting (LightGBM)
- Multi-Layer Perceptron (MLP)
- Soft-Voting Ensemble (equal weighting)

The Ensemble combines predicted probabilities from heterogeneous base learners to stabilise classification performance.

## Evaluation Strategy

Model performance was evaluated using:
- Precision-Recall curves
- Average Precision (AP)
- Evaluation Metrics e.g Confusion Matrices and ROC-AUC comparison

The final soft-voting ensemble achieved the highest Average Precision (0.931) among the evaluated configurations, outperforming individual classifiers trained on both full and reduced feature sets.


## Key
