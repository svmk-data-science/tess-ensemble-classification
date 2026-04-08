# Exoplanet Candidate Classification Using Ensemble Machine Learning on Large-Scale Public Data

*A rigorous comparative study of probabilistic classification models and ensemble methods applied to large, noisy observational data.*


## Executive Summary

This project develops a reproducible ensemble learning framework to improve the automated screening of exoplanet candidates using data derived from Transiting Exoplanet Survey Satellite (TESS) observations available via NASA's Exoplanet Archive.

The objective was to evaluate and compare heterogeneous classification models and assess whether a soft-voting ensemble model improves probabilistic discrimination performance in a large, noisy observational dataset.

Multiple models, which included Random Forest, Support Vector Machine, LightGBM and Multi-Layer Perceptron, were combined into an ensemble model to enhance predictive performance and robustness.

The ensemble model achieved strong results across key metrics with superior discrimination performance (Average Precision = 0.923), while maintaining controlled false-positive rates.

Model interpretability was addressed using SHAP analysis, revealing a small subset of astrophysical features driving the majority of predictions.

This work demonstrates how ensemble methods can improve the priortisation of exoplanet candidates, supporting more effective allocation of follow-up observational resources in large-scale astronomical surveys.

## Problem Statement

The TESS candidate catalogue contains large scale photometric data used to identify potential exoplanets, but candidate classifications are often uncertain due to noisy signals and false positives due to astrophysical and instrumental effects.

This project addresses the challenge of developing a robust probabilistic classification framework to distinguish true planetary signals from non-planetary detections. A key objective is to achieve an effective balance between precision and recall, ensuring that meaningful candidates are identified while minimising false positives.

To achieve this, multiple machine learning models are evaluated and combined using a soft-voting ensemble approach to improve predictive performance, classification stability and precision-recall trade-offs compared to individual model types. The resulting framework is also designed to be computationally efficient and accessible, enabling use by researchers and citizen scientists with limited computational resources.

Although developed in the astrophysical context, the modelling and evaluation framework is transferable to other domains, including fraud detection, risk modelling, anomaly detection and medical diagnosis.

## Dataset

The dataset was obtained from the TESS Project Candidates Catalogue via  NASA Archive (snapshot as at March 2025) and comprised photometric and derived features used to identified exoplanet candidates. A binary target was constructed by encoding candidate disposition as '1' (confirmed) and '0' (false positive) forming the basis for supervised classification. The initial dataset contained 7525 samples with 65 features prior to pre-processing. 


## Project Overview

This project develops an end-to-end machine learning pipeline, which encompasses data pre-processing and cleaning, feature transformation and train-test separation and class balancing for robust model evaluation.

Multiple machine learning models are compared using precision-recall metrics, followed by the implementation of a soft-voting ensemble to improve predictive stability and overall classification performance. Threshold analysis is further applied to assess trade-offs in candidate selection and support effective priortisation.


## Methodology

This project adopts a quantitiative framework guided by the Knowledge Discovery in Database (KDD) methodology, enabling systematic extraction of patterns from large real-world datasets.

### Data Source

The Transiting Exoplanet Survey Satellite (TESS) Project Candidate Catalogue consists of derived planetary and stellar parameters generated from light curve observations, where noise reduction and detrending have already been applied by the upstream pipeline.

### Data Processing & Feature Engineering 

Data pre-processing was conducted to ensure consistency and model readiness. Instances with ambiguous or non-confirmed dispositions (e.g 'FA'.'APC', 'PC' and missing labels) were removed, in addition to features that were missing value or those unsuitable for median imputation. Auxillary limit features did not provide meaningful predictive value and were excluded from the sample. The remaining numerical features were imputed using median values and transformed using logarithmic scaling to reduce skewness and stabilise variance. Data was then scaled using StandardScaler to ensure consistent feature magnitudes. The final dataset reduces to 2054 instances and 22 features, providing a clean and structured input for model development. The dataset was subsequently split into training and testing subsets for model development and evaluation. To address class imbalance, SMOTE (Synthetic Minority Oversampling Technique) applied to the training subset to prevent data leakage. 


### Feature Selection & Dimensionality Reduction

Feature importance was assessed using SHAP (Shapley Additive Explanations) enabling transparent ranking of feature contributions. Features that account for 95% of cumulative importance were retained to reduce dimensionality without significant loss of information.

Principal Component Analysis (PCA) was applied to minimise redundacy and optimise feature representation, improving computational efficiency and model performance.


## Model Training and Evaluation

The following classifiers were implemented as base learners and tuned:
- Support Vector Classifier (SVC)
- Random Forest (RF)
- Gradient Boosting (LightGBM)
- Multi-Layer Perceptron (MLP)
- Soft-Voting Ensemble (equal weighting)

These models were selected for their ability to capture non-linear relationships, handle noisy data and perform effectively on structured datasets.

Model Hyperparameters tuning was performed via Grid Search CV and Randomised Search CV (Pedregosa et al. 2011) with performance evaluated through five-fold cross-validation. 

The ensemble framework adopted a soft voting strategy, aggregating the predicted class probabilities from each base learner, to allow more confident classifiers exert greater influence on the final prediction. 

Model evaluation was based on accuracy, precision, F1-score and Receiver Operating Characteristic (ROC) curves, supported by confusion matrices to quantify trade-offs between sensitivity and specificity. 


## Results/Evaluation

Model evaluation was carried out using multiple complementary metrics, including accuracy, precision, recall, and F1-score, alongside Receiver Operating Characteristic (ROC-AUC) curves and Precision–Recall (PR) curves. Confusion matrices were additionally used to provide a detailed breakdown of classification outcomes, enabling explicit assessment of trade-offs between sensitivity (true positive rate) and specificity (true negative rate).

### Model Performance Comparison

All models exhibit strong discriminative performance, achieving high scores across both ROC-AUC (Figure 1) and Precision-Recall AUC metrics (Figure 2). The ensemble model provides the best overall performance, attaining a ROC-AUC of 0.95 and PR-AUC of 0.931, indicating superior ability to distinguish between confirmed planets and false positives as well as robustness across classification thresholds.

Both the RF and LightGBM models demonstrate strong performance, reflecting the effectiveness of tree-based ensembles in modelling non-linear relationships within structured tabular data. The SVC model also performs competitively, maintaining stable class separation across thresholds. In contrast, the MLP model has lower overall performance and increased variability, particularly at low recall levels, suggesting sensitivity to class imbalance and reduced stability under the given feature representation.

To support operational decision-making, an optimal classification threshold was determined using Youden's J Statistic (J = TPR - FPR). The analysis identified an optimal threshold of approximately 0.464, corresponding to the point on the ROC curve that maximises the trade-off between sensitivity (true positive rate) and specificity (true negative rate). This value offers a balanced operating point, improving detection performance while limiting false positives - an important consideration in exoplanet candidate screening where follow-up validation is resource-intensive.

The ROC curves further confirm that all models significantly outperform the random baseline, with the ensemble model achieving a high true positive rate across most false positive rates. This behaviour indicates improved ranking calibration and reduced variance, reinforcing the advantage of ensemble learning in enhancing both predictive accuracy and robustness.

The confusion matrix (Figure 3) indicates strong overall classification performance of the ensemble model at the optimal threshold (0.46). The model correctly identifies 177 confirmed exoplanets (true positives) and 186 non-confirmed cases (true negatives) with balanced predictive capability across both classes.

Misclassification rates are relatively low with 16 false negatives (3.9%) and 32 false positives (7.8%), the result corresponding to a precision of 0.85, recall of 0.92 and an F1-Score of 0.88. A larger number of false positives compared with false negatives suggests a modest bias towards identifying candidates as confirmed, which is often desirable in exoplanet screening as missing true positives may result in overlooked discoveries, whereas false positives can be filtered through subsequent validation process.

Overall, the results reflect a well-calibrated trade-off between sensitivity and specificity, consistent with the threshold derived using Youden's J Statistic.

### Precision-Recall Curve

<p align="center">
  <img src="images/pr_curve_model_comparison_mixed_features.png" width="450" height="350">
</p>

*Figure 1: Precision–Recall curve for all evaluated models. The ensemble model outperforms all other models by achieving the highest average precision (AP = 0.931) while maintaining superior precision across a broad range of recall levels.*

### ROC Curve

<p align="center">
  <img src="images/ROC_curve_model_comparison.png" width="500">
</p>


*Figure 2: ROC curve comparison across classification models. The optimal decision threshold (≈ 0.464), determined using Youden’s J statistic (J = TPR − FPR), is indicated, representing the point that maximises the trade-off between sensitivity and specificity.*

### Confusion Matrix

<p align="center">
  <img src="images/confusion_matrix_ensemble_publish.png" width="50%">
</p>


*Figure 3: Confusion matrix for the ensemble model at the optimal threshold (≈ 0.46), derived from maximising Youden’s J statistic..*

## Threshold Optimisation

## Model Interpretability
To understand which features most strongly influenced classification decisions, SHAP (Shapely Additive Explanation) values were computed for the ensemble model.

The analysis identifies the most influential features contributing to candidate classification, providing transparency to the ensemble's decision making process.

<p align="center">
  <img src="images/ensemble_SHAP_feature_importance.png" width="120%">
  </p>

*Figure 4: SHAP-based feature importance  (left) and cumulative importance (right) for the ensemble model. Features are ranked by mean absolute SHAP values, highlighting the most influential variables driving classification decisions. The cumulative curve shows that approximately 80% of the model's predictive influence is explained by the top 12 features.*

## Key Insights
The final soft-voting ensemble achieved the highest Average Precision (0.931) among the evaluated configurations, outperforming individual classifiers, particularly LightGBM (AP=0.923) trained on both full and reduced feature sets.

## Reproducibility
This project was developed using Python and widely used machine learning libraries.

Core dependencies are listed in "requirements.txt". To reproduce the environment:

*```bash*
*pip install -r requirements.txt*

The analysis was conducted in Jupyter Notebook. All preprocessing, feature engineering, model training and evaluation steps are documented within the notebooks.


## Evaluation Strategy

Model performance was evaluated using a held-out test set that preserved the original class distribution to ensure unbias. Given the nature of the classification task, evaluation focused on Precision–Recall (PR) curves and Average Precision (AP) rather than accuracy, as PR analysis provides a more informative assessment under class imbalance.

In addition to Precision-Recall analysis, models were evaluated using:
- Receiver Operating Characteristic (ROC) curves and ROC-AUC
- Confusion matrices at selected operating thresholds

ROC-AUC was uses to assess overall ranking performance across thresholds and to compare discriminative capacity independent of class-specific error costs.

Confusion matrices were examined at representative probability thresholds to analyse the trade-off between false positives and false negatives, providing operational insight into screening performance under different decision criteria.

While ROC-AUC offers a global view of separability, analysis of precision degradation at higher recall levels is more informative than true negative rates.

Threshold selection was explored to balance recall sensitivity against precision stability depending on screening objectives.



## Future Work

