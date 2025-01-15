# Healthcare Prediction on Diabetic Patients Using Machine Learning

This repository contains a predictive model developed to identify diabetic patients based on a comprehensive dataset of medical predictor variables. Leveraging data preprocessing, exploratory data analysis (EDA), and machine learning techniques, the project aims to achieve reliable and accurate predictions.

## Dataset Description

The dataset includes various medical predictor variables and one target variable:

### Predictor Variables:
- Pregnancies
- Glucose concentration
- Blood Pressure
- Skin Thickness
- Insulin levels
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

### Target Variable:
- **Outcome**: 1 for diabetic, 0 for non-diabetic.

The dataset contains **768 entries**, with **268 instances labeled as diabetic**.

## Features and Workflow

### 1. **Data Preprocessing**
- Null values were replaced with column means to ensure reliable analysis.
- Outlier detection and treatment were performed using Interquartile Range (IQR) methods.

### 2. **Exploratory Data Analysis**
- Visualizations such as histograms, boxplots, scatter plots, violin plots, and heatmaps were used to understand data distribution and correlations.
- Insights into feature relationships and their potential health implications were drawn.

### 3. **Machine Learning Models**
- Various algorithms were applied:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors (KNN)
- Achieved a **73% accuracy** using Logistic Regression.

### 4. **Model Evaluation**
- Metrics such as accuracy, precision, recall, F1-score, and ROC-AUC were used to assess performance.
- Hyperparameter tuning was applied to improve model efficiency.

## Tools and Libraries
- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`
- **Environment**: Google Colab for seamless execution and visualization.

## Getting Started

### Prerequisites
Install the required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

### **Open in Google Colab** 
You can run this project directly in Google Colab:<br>
Replace your-username, your-repository-name, and your-notebook.ipynb with your GitHub username, repository name, and notebook file name.

### **Execution**
1. **Clone the repository:**
```bash
 git clone \<repository-url\>
```
2. **Navigate to the project directory:**
```bash
 cd diabetes-prediction
```
3. **Run the Jupyter Notebook or Python script in Google Colab or your local environment.** <br>
### **Data**
The dataset (health care diabetes.csv) is included in the repository. Ensure it is placed in the working directory for smooth execution.
## **Results**
Algorithm	and  its Accuracy:
- **Logistic Regression - 73.38%**
- **Random Forest	- 73.38%**
- **Decision Tree -	68.83%**
- **KNN	- 62.34%** <br>
Logistic Regression emerged as the most reliable model, showcasing its strength in binary classification tasks.
