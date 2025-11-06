
---

# ML Challenge 2025: Smart Product Pricing Solution

*Team Name:* The Optimizers  
*Team Members:*  
- Ravinder Singh  
- Raj Verma  
- Ritesh Yadav  
- Ayush Prajapati  

*Submission Date:* 13 October 2025

---

## 1. Executive Summary
Our team’s solution integrates structured product attributes with a tuned gradient boosting framework to predict optimal product prices. By leveraging log-transformations, feature engineering, and Bayesian optimization through Optuna, we achieved strong predictive performance and model stability across validation folds.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
We interpreted the challenge as a regression task, where the objective is to predict a product’s price based on multiple categorical and numerical attributes. Extensive EDA was performed to detect trends, correlations, and feature interactions influencing price.

*Key Observations:*
- Price distribution is right-skewed; log transformation normalizes the target.  
- Product *brand* and *category* are the most influential features.  
- Missing data treatment and encoding methods (Label Encoding + Target Encoding) significantly improved model accuracy.  
- Outlier removal and normalization reduced variance and improved stability.  

### 2.2 Solution Strategy
We implemented a single strong model pipeline using *LightGBM* and tuned its hyperparameters via *Optuna (Bayesian optimization)*. The log-transformed target made the model more robust to skewed price distributions.

*Approach Type:* Single Model (LightGBM Regressor)  
*Core Innovation:* Log-transformed target regression with automated Optuna-based hyperparameter optimization and advanced categorical feature encoding.

---

## 3. Model Architecture

### 3.1 Architecture Overview

## 3. Model Architecture

### 3.1 Architecture Overview

The following diagram illustrates the complete data flow of the product price prediction pipeline — from raw data ingestion to final model output.

```mermaid
flowchart LR

    A[Raw Product Data]

    A --> B[Data Preprocessing]
    B --> B1[Missing Value Imputation]
    B --> B2[Label & Target Encoding]
    B --> B3[Log-Transform Target Variable]

    A --> C[Feature Engineering]
    C --> C1[Derived Product-Level Ratios]
    C --> C2[Quantity-Based Normalization]
    C --> C3[High-Cardinality Feature Handling]

    A --> D[Model Training]
    D --> D1[LightGBM Regressor]
    D --> D2[Optuna Bayesian Tuning]

    D --> E[Validation & Early Stopping]

    E --> F[Prediction]
    F --> F1[Inverse Log-Transform → Final Price]




### 3.2 Model Components

*Text Processing Pipeline:*
- [x] Preprocessing steps: lowercasing, cleaning special characters, label encoding categorical text features  
- [x] Model type: LightGBM Regressor  
- [x] Key parameters (optimized using Optuna):  
  - num_leaves  
  - max_depth  
  - learning_rate  
  - feature_fraction  
  - bagging_fraction  
  - min_data_in_leaf  
  - lambda_l1, lambda_l2

*Image Processing Pipeline:*
- [ ] Preprocessing steps: N/A  
- [ ] Model type: N/A  
- [ ] Key parameters: N/A

---

## 4. Model Performance

### 4.1 Validation Results
- *SMAPE Score:* 52.89  
- *Other Metrics:*  
  - MAE: 152.8  
  - RMSE: 230.3  
  - R²: 0.89  

---

## 5. Conclusion
Our solution effectively models complex non-linear dependencies between product features and price using an optimized LightGBM pipeline. The log-transformation and Optuna-based tuning contributed significantly to the model’s accuracy and generalization. Future enhancements could include multimodal learning (e.g., image-based product representation) and price trend tracking for dynamic updates.

---

## Appendix

### A. Code Artefacts

*Code Directory Structure:*

📁 Smart_Product_Pricing/ │ ├── data/ │   ├── train.csv │   ├── test.csv │ ├── notebooks/ │   ├── EDA.ipynb │ ├── src/ │   ├── preprocess.py │   ├── feature_engineering.py │   ├── train.py │   ├── optuna_tuning.py │   ├── inference.py │ ├── models/ │   ├── lgbm_best_model.pkl │ └── README.md

---

### B. Additional Results
- *Feature Importance:* Brand, Category, and Quantity emerged as dominant features.  
- *Training Time:* ~2.8 minutes per Optuna trial on GPU (Colab environment).  
- *Validation Strategy:* 5-Fold Stratified Split for better generalization.  
- *Deployment Readiness:* Model serialized with pickle for direct API integration.

---
