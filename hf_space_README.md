---
title: Wine Color Classifier
emoji: 🍷
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Wine Color Classifier + Adversarial Analysis

Classify red vs. white wine from 11 chemical features — with live SHAP explanations
and adversarial perturbation analysis.

**Models:** Logistic Regression (F1: 0.9938) · XGBoost (ROC-AUC: 0.9999)

**Adversarial twist:** after classifying your input, the app computes the minimum
chemical change (FGSM, L∞ norm) that would flip the result — and shows you exactly
which features to tweak, and by how much.
