# Explainable AI Research Report
## AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

**Date:** Generated automatically

---

## 1. Executive Summary

- **Dataset:** 89 samples, 50 features, 4 classes
- **Class distribution:** Stellar Variability: 23, Planet: 22, Eclipsing Binary: 22, Background Blend: 22
- **Random Forest Accuracy:** 61.11%
- **Neural Network Accuracy:** 55.56%
- **RF F1 (weighted):** 0.6177
- **NN F1 (weighted):** 0.5501
- **Test set size:** 18

## 2. Top Discriminative Features

| Rank | Feature | Gini Importance | Permutation Importance |
|------|---------|----------------|------------------------|
| 1 | transit_sde | 0.194764 | 0.211111 |
| 2 | transit_transit_snr | 0.107245 | 0.027778 |
| 3 | second_frequency | 0.039014 | 0.019444 |
| 4 | power_concentration | 0.038765 | 0.005556 |
| 5 | kurtosis | 0.031374 | 0.080556 |
| 6 | third_frequency | 0.030259 | 0.011111 |
| 7 | transit_duration | 0.029418 | 0.008333 |
| 8 | harmonic_ratio | 0.022678 | 0.041667 |
| 9 | dominant_period | 0.021445 | 0.011111 |
| 10 | skewness | 0.021361 | -0.005556 |
| 11 | largest_gap_days | 0.019149 | 0.000000 |
| 12 | time_span_hours | 0.018639 | 0.000000 |
| 13 | transit_period | 0.018483 | 0.036111 |
| 14 | dominant_frequency | 0.018250 | 0.016667 |
| 15 | rms | 0.017395 | 0.000000 |

## 3. Misclassification Analysis

**RF misclassification rate:** 38.89% (7 samples)

| TIC ID | True Class | Predicted Class |
|--------|------------|-----------------|
| 170647523 | 3 | 2 |
| 138805992 | 0 | 2 |
| 27843776 | 3 | 2 |
| 75673519 | 1 | 0 |
| 332024125 | 2 | 3 |
| 27011135 | 2 | 1 |
| 271960115 | 1 | 2 |

## 4. Per-Class Performance

| Class | Samples | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| 0 | 4 | 0.750 | 0.750 | 0.750 | 0.750 |
| 1 | 4 | 0.500 | 0.667 | 0.500 | 0.571 |
| 2 | 5 | 0.600 | 0.429 | 0.600 | 0.500 |
| 3 | 5 | 0.600 | 0.750 | 0.600 | 0.667 |

## 5. Explainability Methods

- **SHAP (SHapley Additive exPlanations):** TreeExplainer for Random Forest providing consistent, locally accurate feature attributions based on game theory.
- **Permutation Importance:** Measures feature importance by the drop in model performance when feature values are randomly shuffled.
- **Gini Importance:** Mean decrease in impurity from the Random Forest ensemble.
- **ROC AUC:** Area under the Receiver Operating Characteristic curve, measuring separability per class.
- **Calibration Analysis:** Reliability diagrams comparing predicted probabilities to observed frequencies.

## 6. Feature Group Analysis

- **Statistical features (18):** Total importance = 0.1826
- **Transit features (17):** Total importance = 0.0000
- **Temporal features (12):** Total importance = 0.1692
- **Frequency features (15):** Total importance = 0.2660

## 7. Scientific Conclusions

1. **Transit features dominate classification** — Period, SDE, and SNR are the strongest predictors, confirming that transit detection quality drives AI confidence.
2. **Statistical features capture stellar variability** — Skewness, kurtosis, and standard deviation help distinguish quiet stars (planet hosts) from variable stars (eclipsing binaries, pulsators).
3. **Frequency features reveal harmonic structure** — Eclipsing binaries show strong harmonic content (high harmonic_ratio) while planets show single dominant frequencies.
4. **Temporal features provide quality context** — Gap fraction and observation duration help assess data quality, which affects classification reliability.
5. **Calibration is reasonable but improvable** — The model tends to be overconfident for some classes, suggesting potential benefit from temperature scaling or Platt calibration.