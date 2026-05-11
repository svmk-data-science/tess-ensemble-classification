# 🧠 Exoplanet Candidate Classification Using Ensemble Learning
A high-performance ensemble model achieving PR-AUC of 0.93 for exoplanet candidate classification using interpretable machine learning.

🔬 Machine Learning | 🌌 Astrophysics | 🤖 Ensemble Models | 📊 Explainable AI

---

## 🚀 Overview
Exoplanet detection pipelines often produce high false-positive rates due to noisy and imbalanced observational data.

This project develops an ensemble machine learning framework to classify exoplanet candidates using data from the NASA Exoplanet Archive.

The goal is to improve prediction reliability and reduce false positives in these challenging conditions.

---

## 📊 Key Results
- **Average Precision (PR-AUC):** 0.931  
- **ROC-AUC:** 0.95  
- **Precision:** 0.85  
- **Recall:** 0.92  
- **Optimal Threshold:** ~0.46  

---

## 🛠 Tech Stack
- Python  
- Scikit-learn, LightGBM  
- Pandas, NumPy  
- SHAP (Explainable AI)  
- Matplotlib, Seaborn  

---

## ⚙️ Approach
**Example improvement:**
- Data preprocessing, cleaning and feature engineering
- Train/test split to prevent data leakage
- Model benchmarking (SVM, Random Forest, LightGBM, MLP)
- Soft-voting ensemble model construction
- Precision–Recall evaluation for imbalanced data
- Threshold optimisation (Youden’s J statistic)


---

## 💡 Key Insights
- Ensemble learning reduced variance and improved stability across candidate classification
- Precision–Recall optimisation proved more informative than ROC-based evaluation under class imbalance
- SHAP analysis revealed that a small subset of astrophysical features dominates model decision-making
- Threshold tuning enables configurable trade-offs between false positives and missed detections  
  
---

## 📄 Full Technical Report
For detailed methodology, evaluation, and discussion:

👉 [View Full Report](FULL_REPORT.md)

---

## 📌 Project Highlights
- End-to-end machine learning pipeline from preprocessing to evaluation  
- Interpretable AI applied to scientific data  
- Designed with a focus on practical deployment and interpretability  

---

## 💻 Code Samples

Notebooks have been streamlined to highlight key modelling components.
Some preprocessing and engineering steps have been intentionally condensed or omitted for brevity.

The full methodology and pipeline design are documented in FULL_REPORT.md.

- ensemble_model_evaluation.ipynb  
- shap_feature_analysis.ipynb  

---

## 🏁 Summary
This project demonstrates how ensemble learning and explainable AI can be combined to deliver accurate, transparent, and practical solutions for exoplanet candidate classification.

