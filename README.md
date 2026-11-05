# 📊 Job Acceptance Prediction System

An end-to-end Data Science and Machine Learning project that analyzes candidate placement data and predicts placement outcomes using machine learning techniques.

The project focuses on understanding the factors associated with candidate placement, evaluating candidate employability, and providing recruitment insights through an interactive Streamlit dashboard.

## 📌 Project Overview

Recruitment and placement decisions involve multiple factors, including academic performance, technical skills, interview performance, work experience, and job-role suitability.

This project uses historical candidate data to:

- Analyze candidate placement patterns.
- Identify factors associated with successful placement.
- Perform data cleaning and exploratory data analysis.
- Engineer meaningful features for machine learning.
- Build and evaluate classification models.
- Predict candidate placement outcomes.
- Present recruitment insights through a Streamlit dashboard.

**Domain:** HR Analytics / Recruitment Analytics

**Project Type:** End-to-End Data Science and Machine Learning

## 🎯 Project Objectives

1. Analyze candidate academic and professional performance.
2. Identify factors associated with successful placement.
3. Handle missing values, duplicate records, and inconsistent categorical data.
4. Perform exploratory data analysis (EDA).
5. Develop meaningful features for predictive modeling.
6. Build and evaluate machine learning classification models.
7. Develop an interactive recruitment analytics dashboard.

## 📂 Dataset Information

The project uses a synthetic HR Job Placement dataset containing 51,500 original records and 26 columns.

After data cleaning and duplicate removal, the final dataset contains **50,000 candidate records**.

### Dataset Features

| Category | Features |
|---|---|
| Personal Information | Age, Gender |
| Academic Performance | SSC, HSC, Degree Percentage, Degree Specialization |
| Skills & Assessment | Technical Score, Aptitude Score, Communication Score, Skills Match Percentage |
| Experience | Years of Experience, Relevant Experience, Internship Experience |
| Recruitment Factors | Company Tier, Job Role Match, Competition Level |
| Other Factors | Certifications, Previous CTC, Expected CTC, Notice Period, Relocation Willingness |
| Target Variable | Status |

**Target variable:**
- `Placed` → 1
- `Not Placed` → 0

## 🧹 Data Cleaning and Preprocessing

The following data preparation steps were performed:

- Identified and handled missing values.
- Applied median imputation to numerical columns.
- Applied categorical imputation using an `Unknown` category where appropriate.
- Standardized inconsistent categorical values.
- Removed duplicate records.
- Checked numerical ranges and potential data inconsistencies.
- Encoded categorical variables.
- Scaled numerical features.
- Used stratified train-test splitting.

Final cleaned dataset:

- Records: 50,000
- Original columns: 26
- Engineered dataset columns: 33
- Remaining missing values: 0
- Remaining duplicate records: 0

## 📊 Exploratory Data Analysis (EDA)

The project explored relationships between candidate characteristics and placement outcomes.

Key analyses included:

- Placement outcome distribution.
- Technical, aptitude, and communication scores by placement status.
- Skills match percentage and placement outcomes.
- Company tier versus placement rate.
- Experience category versus placement success.
- Competition level versus placement outcomes.
- Correlation analysis of numerical features.

### Key Findings

- Overall placement rate: **30.26%**
- Not placed candidates: **69.74%**
- Placed candidates showed a higher average technical score.
- Placed candidates had a higher average skills match percentage.
- Experienced candidates showed a higher placement rate than freshers in this dataset.
- Company tier showed relatively small differences in placement rates.

These findings represent associations in the dataset and do not establish causal relationships.

## ⚙️ Feature Engineering

The following features were created:

| Feature | Description |
|---|---|
| `experience_category` | Categorizes candidates based on years of experience |
| `academic_average` | Average of SSC, HSC, and degree percentages |
| `academic_performance_band` | Categorizes academic performance |
| `skills_match_level` | Categorizes skills match percentage |
| `interview_average` | Average of technical, aptitude, and communication scores |
| `interview_performance_category` | Categorizes interview performance |
| `placement_analytical_score` | Weighted analytical score based on candidate performance |

The `placement_analytical_score` was excluded from the machine learning model to avoid using a manually engineered outcome-related score as a model input.

## 🤖 Machine Learning

### Problem Type

Supervised Machine Learning – Binary Classification.

### Algorithms Evaluated

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting Classifier

### Selected Model

**Gradient Boosting Classifier**

The final model was trained using a preprocessing and classification pipeline.

The preprocessing steps include:

- Numerical imputation and standardization.
- Categorical imputation and one-hot encoding.
- Ordinal encoding for competition level.
- Gradient Boosting classification.

### Model Evaluation

The final leakage-safe evaluation was performed using an 80:20 stratified train-test split and five-fold cross-validation.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 90.58% |
| Precision | 85.30% |
| Recall | 83.21% |
| F1 Score | 84.24% |
| ROC-AUC | 96.79% |

**Cross-validation F1 Score:** 84.60% (mean across five folds).

The model was evaluated using multiple metrics because the target classes are imbalanced.

## 📈 Streamlit Dashboard

An interactive dashboard was developed using Streamlit to present recruitment analytics and placement insights.

### Dashboard KPIs

- Total Candidates
- Placement Rate
- Job Acceptance Rate
- Average Interview Score
- Average Skills Match
- Offer Dropout Rate
- High-Risk Candidates

### Analytical Questions

The dashboard is designed around seven analytical questions:

1. Academic Scores vs Placement Outcome
2. Skills Match vs Interview Performance
3. Certification Impact on Job Acceptance
4. Acceptance Rate by Company Tier
5. Experience vs Placement Success
6. Interview Score vs Placement Probability
7. Employability Test Score Analysis

The dashboard currently includes the initial analytical question and the remaining analyses are planned for implementation.

**Data limitation:** The available dataset contains placement status, not verified job-offer acceptance or dropout records. Therefore, job acceptance rate and offer dropout rate cannot currently be calculated reliably. Placement rate is used as the available outcome measure.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and preprocessing |
| SQLAlchemy | Database connectivity |
| MySQL | Data storage |
| Jupyter Notebook | Data analysis and experimentation |
| VS Code | Development environment |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualization |
| Joblib | Model serialization |

## 📁 Project Structure

```text
Job-Acceptance-Prediction-System/
│
├── data/
│   ├── HR_Job_Placement_Dataset.csv
│   └── HR_Job_Placement_Cleaned.csv
│
├── models/
│   └── final_pipeline.pkl
│
├── job_acceptance_analysis.ipynb
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd Job-Acceptance-Prediction-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not available, install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sqlalchemy pymysql streamlit plotly joblib
```

### 4. Run the Jupyter Notebook

Open the notebook:

```text
job_acceptance_analysis.ipynb
```

Run the cells to explore the data analysis and machine learning workflow.

### 5. Launch the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

## 💾 Model Information

The trained machine learning pipeline is saved as:

```text
models/final_pipeline.pkl
```

The pipeline includes the preprocessing transformations and trained Gradient Boosting classifier.

The application loads the saved pipeline using Joblib.

**Important:** Load model files only from trusted sources. Scikit-learn and Joblib model files may execute code during deserialization.

## 📌 Future Improvements

- Complete all seven analytical dashboard sections.
- Implement model-based candidate placement probability predictions.
- Calculate high-risk candidate percentages using a documented probability threshold.
- Add an interactive candidate-level prediction interface.
- Improve model interpretability using feature importance and SHAP.
- Add verified offer acceptance and dropout data.
- Deploy the dashboard to a cloud platform.
- Improve model monitoring and evaluation.

## ⚠️ Limitations

- The dataset contains placement outcomes rather than verified job-offer acceptance outcomes.
- The model's predictions depend on the quality and representativeness of the available dataset.
- Model predictions indicate estimated placement likelihood, not guaranteed recruitment outcomes.
- Relationships identified through EDA should not be interpreted as causal effects.
- Dashboard development is ongoing.

## 👨‍💻 Author

**Azeem Basha**

Data Science and Machine Learning Enthusiast

Skills: Python | SQL | Data Analysis | Machine Learning | Power BI | Streamlit

---

## ⭐ Acknowledgement

This project was developed as part of the GUVI Data Science with Generative AI learning journey.

Feedback and suggestions are welcome!
