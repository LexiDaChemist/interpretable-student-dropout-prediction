# Interpretable Student Dropout Prediction

This project applies **interpretable machine learning** techniques to predict student academic outcomes—**Dropout**, **Enrolled**, or **Graduate**—using demographic, academic, and financial indicators.  
The primary focus is on **model transparency, evaluation, and responsible interpretation**, rather than maximizing predictive performance alone.

The final model is a **multiclass logistic regression** pipeline built with scikit-learn and deployed with explanatory visualizations and scenario-based predictions.

---

## Project Goals

- Predict student academic outcomes using an interpretable model  
- Analyze how academic progress and financial stress influence predictions  
- Provide **transparent, conditional explanations** of model behavior  
- Demonstrate responsible use of predictive modeling in an educational context  

---

## Modeling Approach

- **Model:** Multiclass Logistic Regression (scikit-learn)
- **Preprocessing:**  
  - Median imputation for numerical variables  
  - Mode imputation + one-hot encoding for categorical variables  
- **Class imbalance:** Handled via `class_weight="balanced"`
- **Evaluation:**  
  - Held-out test set  
  - Confusion matrix and class-wise metrics  
- **Explainability:**  
  - Coefficient-based feature influence plots (per class)  
  - Distinction between descriptive outcome patterns and conditional model effects  
- **Prediction Demo:**  
  - Scenario-based predictions (not a live risk calculator)  

---

## Dataset

**Dataset name:** *Predict Students’ Dropout and Academic Success*  
**Platform:** Kaggle  

This dataset includes demographic attributes, academic performance indicators, and financial status variables collected from higher education students.

### Kaggle Contributors

The Kaggle dataset was curated and shared by the following contributors:
- **The Devastator**  
- **CarmelH**  
- **Sean Mauer**

Their work enabled broader access to the dataset for educational and research purposes.

---

## Original Data Source (Academic Attribution)

The dataset originates from the following peer-reviewed research article:

> Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022).  
> *Predicting student dropout and academic success*.  
> **Education Sciences**, 12(4), 276.  
> https://doi.org/10.3390/educsci12040276

All credit for data collection, study design, and original analysis belongs to the above authors.  
Kaggle distributions of the dataset trace back to this publication.

---

## Responsible Use & Limitations

- This project uses **observational data**; results should not be interpreted as causal.
- Predictions are intended to support **analysis and outreach**, not punitive or automated decision-making.
- Scenario-based predictions are included to demonstrate model behavior responsibly, avoiding individual risk scoring.
- Fairness, bias, and contextual evaluation are essential before any real-world deployment.

---

## Current Status

- ✔ Data ingestion and preprocessing pipeline complete  
- ✔ Multiclass logistic regression model trained  
- ✔ Evaluation and explainability visualizations generated  
- ✔ Scenario-based predictions implemented  
- ✔ Project website deployed via GitHub Pages  

---

## Future Work

- Fairness and subgroup performance analysis  
- Temporal modeling of academic progression  
- Comparison with other interpretable models (e.g., GAMs)  
- Robustness and stability analysis of model coefficients  

---

## License & Attribution

This repository is intended for **educational and research purposes only**.  
Please cite the original publication if using the dataset or derived results in academic work.

## Repository Structure

