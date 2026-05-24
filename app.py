import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# ── Load models ────────────────────────────────────────────────────────────────
lr_pipeline  = joblib.load("models/lr_pipeline.pkl")
xgb_pipeline = joblib.load("models/xgb_pipeline.pkl")

FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]

# Pre-compute SHAP explainer and LR internals once at startup
_xgb_model = xgb_pipeline.named_steps["xgb"]
_explainer  = shap.TreeExplainer(_xgb_model)
_scaler     = lr_pipeline.named_steps["scaler"]
_lr         = lr_pipeline.named_steps["lr"]
_w          = _lr.coef_[0]
_b          = float(_lr.intercept_[0])
_w_l1       = float(np.abs(_w).sum())

# ── Slider config: (min, max, step, avg_red, avg_white) ────────────────────────
FEATURE_CONFIG = [
    ("fixed acidity",        3.8,   15.9,  0.1,    8.32,  6.85),
    ("volatile acidity",     0.08,   1.58, 0.01,   0.528, 0.278),
    ("citric acid",          0.0,    1.66, 0.01,   0.271, 0.334),
    ("residual sugar",       0.6,   65.8,  0.1,    2.54,  6.39),
    ("chlorides",            0.009,  0.611,0.001,  0.0875,0.0458),
    ("free sulfur dioxide",  1.0,  289.0,  1.0,   15.87, 35.31),
    ("total sulfur dioxide", 6.0,  440.0,  1.0,   46.47,138.36),
    ("density",              0.987,  1.039,0.0001, 0.9967,0.9940),
    ("pH",                   2.72,   4.01, 0.01,   3.311, 3.188),
    ("sulphates",            0.22,   2.0,  0.01,   0.658, 0.490),
    ("alcohol",              8.0,   14.9,  0.1,   10.42, 10.51),
]

AVG_RED   = [cfg[3] for cfg in FEATURE_CONFIG]
AVG_WHITE = [cfg[4] for cfg in FEATURE_CONFIG]

# A borderline wine — close to the decision boundary for a dramatic demo
BORDERLINE = [7.8, 0.40, 0.30, 5.5, 0.062, 28.0, 98.0, 0.9952, 3.24, 0.56, 10.45]


# ── Core inference function ────────────────────────────────────────────────────
def classify_wine(*features):
    x    = np.array(features, dtype=float).reshape(1, -1)
    x_df = pd.DataFrame(x, columns=FEATURE_COLS)

    lr_pred  = int(lr_pipeline.predict(x)[0])
    lr_prob  = lr_pipeline.predict_proba(x)[0]
    xgb_pred = int(xgb_pipeline.predict(x)[0])
    xgb_prob = xgb_pipeline.predict_proba(x)[0]

    # Adversarial analysis via LR decision function
    x_scaled = _scaler.transform(x)
    z        = float(x_scaled @ _w + _b)
    eps_min  = (abs(z) + 1e-8) / _w_l1
    sign_flip    = -np.sign(z)
    delta_scaled = sign_flip * eps_min * np.sign(_w)
    delta_orig   = delta_scaled * _scaler.scale_

    # SHAP values for this specific input
    sv = _explainer.shap_values(x_df)[0]

    # ── Result card ────────────────────────────────────────────────────────────
    label      = "🍷  RED" if lr_pred == 1 else "🥂  WHITE"
    confidence = lr_prob[lr_pred] * 100
    flip_to    = "White" if lr_pred == 1 else "Red"

    if eps_min < 0.10:
        robustness = "⚠️ Very close to boundary — easily fooled"
    elif eps_min < 0.35:
        robustness = "🟡 Moderate distance from boundary"
    else:
        robustness = "✅ Far from boundary — robust"

    result_md = f"""## {label} &nbsp;·&nbsp; {confidence:.1f}% confidence

| Model | Prediction | Confidence |
|---|---|---|
| Logistic Regression | {"🍷 Red" if lr_pred == 1 else "🥂 White"} | {lr_prob[lr_pred]*100:.1f}% |
| XGBoost | {"🍷 Red" if xgb_pred == 1 else "🥂 White"} | {xgb_prob[xgb_pred]*100:.1f}% |

**Boundary distance:** ε = `{eps_min:.4f}` &nbsp; {robustness}
"""

    # ── Adversarial recipe ─────────────────────────────────────────────────────
    top_idx = np.argsort(np.abs(delta_orig))[::-1][:5]
    lines = [f"### Minimum perturbation to flip this wine → {flip_to}\n"]
    for i in top_idx:
        arrow = "▲" if delta_orig[i] > 0 else "▼"
        lines.append(f"- {arrow} **{FEATURE_COLS[i]}**: `{delta_orig[i]:+.4f}`")
    recipe_md = "\n".join(lines)

    # ── Chart: SHAP + adversarial delta side by side ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    shap_order  = np.argsort(np.abs(sv))
    shap_colors = ["#8B0000" if v > 0 else "#336699" for v in sv[shap_order]]
    axes[0].barh(
        [FEATURE_COLS[i] for i in shap_order], sv[shap_order],
        color=shap_colors, edgecolor="black", linewidth=0.4
    )
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("What drove this prediction\n(XGBoost SHAP)", fontweight="bold")
    axes[0].set_xlabel("← pushes White  |  pushes Red →")

    adv_order  = np.argsort(np.abs(delta_orig))
    adv_colors = ["#8B0000" if v > 0 else "#336699" for v in delta_orig[adv_order]]
    axes[1].barh(
        [FEATURE_COLS[i] for i in adv_order], delta_orig[adv_order],
        color=adv_colors, edgecolor="black", linewidth=0.4
    )
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title(
        f"Minimum chemical change to flip → {flip_to}\n(ε = {eps_min:.4f})",
        fontweight="bold"
    )
    axes[1].set_xlabel("Required Δ in original units")

    plt.tight_layout()

    return result_md, fig, recipe_md


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Wine Color Classifier + Adversarial Analysis",
    theme=gr.themes.Base(),
    css=".gr-button-primary { background: #8B0000 !important; }"
) as demo:

    gr.Markdown("""
# 🍷 Wine Color Classifier + Adversarial Analysis

Adjust the chemical properties below → classify **Red or White**, see the **SHAP explanation**,
and discover the **minimum perturbation** that would fool the model.

*Models: Logistic Regression (F1 0.9938) · XGBoost (ROC-AUC 0.9999) trained on UCI Wine Quality (6,497 samples)*
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Wine Chemistry Inputs")
            sliders = []
            for name, mn, mx, step, avg_r, avg_w in FEATURE_CONFIG:
                s = gr.Slider(minimum=mn, maximum=mx, step=step, value=avg_r, label=name)
                sliders.append(s)

            with gr.Row():
                red_btn      = gr.Button("🍷 Avg Red Wine",    variant="secondary")
                white_btn    = gr.Button("🥂 Avg White Wine",  variant="secondary")
                border_btn   = gr.Button("⚠️ Borderline Wine", variant="secondary")

            classify_btn = gr.Button("Classify →", variant="primary", size="lg")

        with gr.Column(scale=2):
            result_md = gr.Markdown("*Load a preset or adjust sliders, then hit Classify.*")
            chart     = gr.Plot(label="SHAP · Adversarial Perturbation")
            recipe_md = gr.Markdown()

    # Wire up buttons
    classify_btn.click(fn=classify_wine, inputs=sliders, outputs=[result_md, chart, recipe_md])
    red_btn.click(fn=lambda: AVG_RED,    inputs=[], outputs=sliders)
    white_btn.click(fn=lambda: AVG_WHITE, inputs=[], outputs=sliders)
    border_btn.click(fn=lambda: BORDERLINE, inputs=[], outputs=sliders)

    gr.Markdown("""
---
**How the adversarial analysis works:** the LR model's decision boundary is a hyperplane —
`z = w · x_scaled + b`. The minimum L∞ perturbation to cross it is `ε_min = |z| / ‖w‖₁`.
The per-feature delta (right chart) shows exactly which chemicals need to change,
and by how much, to flip the classification. Small ε = you're near the boundary.
""")

if __name__ == "__main__":
    demo.launch()
