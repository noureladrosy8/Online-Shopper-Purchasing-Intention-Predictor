# 🛒 Online Shopper Purchasing Intention Predictor

An end-to-end Machine Learning classification project designed to analyze online e-commerce user session behaviors and predict customer purchasing intent (`Revenue`). This project covers the full data science lifecycle—from exploratory data analysis (EDA) and data preprocessing to feature engineering, class imbalance handling, and multi-model benchmarking using the benchmark **Online Shoppers Purchasing Intention Dataset**.

---

## 🌟 Key Features

* **Real-Time Purchasing Intent Classification:** Predicts whether an e-commerce visitor will complete a purchase session or exit without buying.
* **In-Depth Session Analytics:** Comprehensive visual analysis exploring session page counts, browsing duration, seasonal trends, and exit/bounce rates.
* **Robust Preprocessing Pipeline:** Handles skewed numerical distributions, encodes session categorical metadata, and prepares scaled representations for machine learning estimators.
* **Class Imbalance Management:** Implements techniques (e.g., class weighting, re-sampling) to handle minority class detection for actual revenue-generating sessions.
* **Multi-Model Benchmarking:** Evaluates linear and ensemble models using classification metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## 🛠️ Tech Stack & Libraries

* **Programming Language:** Python 3.x
* **Data Manipulation & Processing:** `Pandas`, `NumPy`
* **Exploratory Data Analysis & Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning & Evaluation:** `Scikit-Learn` (Logistic Regression, Random Forest, Metrics, Preprocessing)
* **Model Serialization:** `Pickle` / `Joblib`
* **Dataset Source:** UCI Machine Learning Repository / Kaggle (*Online Shoppers Purchasing Intention Dataset*)

---

## 📂 Project Structure

```text
Online-Shopper-Purchasing-Intention/
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Exploratory Data Analysis & Modeling Notebooks
├── src/                   # Source scripts (preprocessing, training, evaluation)
├── models/                # Serialized trained model checkpoints (.pkl)
└── README.md              # Comprehensive project documentation
```

---

## 🧹 Data Preprocessing & Pipeline

E-commerce web log sessions contain highly skewed distributions and session indicators that require structured cleaning and scaling:

1. **Handling Missing & Invalid Values:**
   * Checked for missing entries and verified numerical integrity across duration and rate metrics.

2. **Feature Transformation & Skewness Reduction:**
   * Applied log transformations or standard scaling on highly skewed metrics like `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration`.

3. **Handling Class Imbalance:**
   * The target feature `Revenue` is highly imbalanced (~84.5% False vs ~15.5% True). Applied techniques such as `class_weight='balanced'` in estimators and SMOTE/undersampling during training to avoid biased predictions toward non-buyers.

4. **Categorical Encoding & Scaling:**
   * **One-Hot Encoding:** Applied to temporal and metadata features (`Month`, `VisitorType`, `Weekend`).
   * **Standardization:** Normalized continuous features (`BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`) using `StandardScaler` to optimize linear and gradient convergence.

---

## 📊 Dataset & Feature Overview

The benchmark dataset consists of 12,330 sessions with 18 distinct features capturing user interaction and session contextual data:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Administrative / Duration** | Numerical | Number of administrative pages visited and time spent (seconds) |
| **Informational / Duration** | Numerical | Number of informational pages visited and time spent (seconds) |
| **ProductRelated / Duration** | Numerical | Number of product pages visited and time spent (seconds) |
| **BounceRates** | Numerical | Percentage of visitors entering the site and leaving without further requests |
| **ExitRates** | Numerical | Percentage of page views that were the last in the session |
| **PageValues** | Numerical | Average value of pages visited prior to completing a transaction |
| **SpecialDay** | Numerical | Closeness of site visit time to a specific holiday/special day |
| **Month / OperatingSystems / Browser / Region** | Categorical | Temporal and environmental user session attributes |
| **VisitorType** | Categorical | Returning Visitor, New Visitor, or Other |
| **Weekend** | Boolean | Indicator if the session occurred on a weekend |
| **Revenue (Target)** | Binary | Class indicator: `True` (Purchased) or `False` (Did not purchase) |

---

## 🔄 Exploratory Data Analysis & Insights

* **Page Value Impact:** `PageValues` emerged as the single strongest predictor of purchase conversion; sessions with `PageValues > 0` exhibited significantly higher conversion rates.
* **Duration vs. Conversion:** Users spending balanced time across `ProductRelated` pages with lower `ExitRates` are far more likely to finalize a checkout.
* **Seasonal & Special Day Variations:** High session volumes in months like November and May show distinct purchasing behavior influenced by sales events, though `SpecialDay` proximity alone doesn't guarantee higher conversions without high engagement.
* **Visitor Loyalty:** Returning visitors account for the bulk of overall traffic, but new visitors often display higher purchase efficiency per session.

---

## 📈 Model Training & Evaluation Results

Multiple classification algorithms were benchmarked to maximize purchase intention detection:

* **Logistic Regression:** Serves as a strong interpretable baseline model.
* **Random Forest Classifier:** Captures complex non-linear feature interactions and high-dimensional session interactions.

### Performance Summary
* **Accuracy Range:** Achieved **88% – 94%** accuracy across test split evaluations.
* **Evaluation Focus:** Evaluated using **ROC-AUC**, **F1-Score**, and **Precision-Recall** metrics to ensure reliable detection of actual buyers without overwhelming false positives.

---

## 👤 Author

**Nour Saad Eladrosy**  
* GitHub: [@noureladrosy8](https://github.com/noureladrosy8)
