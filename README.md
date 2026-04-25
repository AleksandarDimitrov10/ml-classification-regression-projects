# Machine Learning Classification and Regression Projects

This repository contains two supervised machine learning projects built with Python and scikit-learn:

1. Turtle age prediction using regression models
2. Bot detection using classification models

The projects demonstrate data preprocessing, exploratory data analysis, model training, validation, hyperparameter tuning, cross-validation, error analysis, and model evaluation.

## Project 1: Turtle Age Prediction

The first project predicts turtle age using biological and shell measurement features. The target variable is `Rings`, which is treated as a continuous regression target.

### Main Steps

- Loaded and explored the turtle dataset
- Visualized relationships between shell measurements and number of rings
- Computed summary statistics and feature correlations
- Analysed the influence of sex/gender category on age
- Split the data into train, validation, and test sets
- Used target binning to preserve target distribution during splitting
- Trained and evaluated multiple regression models
- Compared Linear Regression, Polynomial Regression, and KNN Regression
- Used 10-fold cross-validation for model stability analysis
- Investigated over-prediction and under-prediction errors
- Saved model comparison results to CSV

### Models Used

- Linear Regression
- Polynomial Regression
- K-Nearest Neighbors Regression

### Evaluation Metrics

- R² score
- Mean Squared Error
- Mean Absolute Error

## Project 2: Bot Detection Classification

The second project classifies accounts as bots or human users using a supervised machine learning pipeline.

### Main Steps

- Loaded and prepared a bot-vs-user dataset
- Split the data into training and testing sets using stratification
- Built preprocessing pipelines for numerical and categorical features
- Imputed missing values
- Scaled numerical features
- One-hot encoded categorical features
- Trained Logistic Regression and Support Vector Machine classifiers
- Used RandomizedSearchCV for hyperparameter tuning
- Evaluated models using classification metrics
- Generated confusion matrices, ROC curves, and precision-recall curves
- Analysed feature importance from Logistic Regression coefficients
- Compared error rates across gender groups

### Models Used

- Logistic Regression
- Support Vector Machine

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Confusion Matrix
- Precision-Recall Curve

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Project Structure

ml-classification-regression-projects/
- README.md
- requirements.txt
- .gitignore
- src/
  - turtle_age_regression.py
  - bot_detection_classification.py
- data/
  - README.md
- results/
  - README.md

## How to Run

### 1. Clone the repository

git clone https://github.com/AleksandarDimitrov10/ml-classification-regression-projects.git

cd ml-classification-regression-projects

### 2. Install dependencies

pip install -r requirements.txt

### 3. Add the datasets

Place the datasets inside the `data/` folder:

- data/turtle-2.csv
- data/bots_vs_users.csv

The scripts expect the datasets to be stored at those paths.

### 4. Run the turtle age regression project

python src/turtle_age_regression.py

### 5. Run the bot detection classification project

python src/bot_detection_classification.py

## My Contribution

This was a machine learning coursework project focused on applying supervised learning methods to both regression and classification tasks.

My contribution involved implementing the full machine learning workflow: data loading, exploratory data analysis, preprocessing, train-validation-test splitting, model training, hyperparameter selection, cross-validation, evaluation, visualization, and error analysis.

For the regression task, I compared Linear Regression, Polynomial Regression, and KNN Regression for turtle age prediction. For the classification task, I built preprocessing pipelines and compared Logistic Regression and SVM models for bot detection.

I also interpreted model performance using appropriate metrics, including R², MSE, MAE, precision, recall, F1-score, ROC AUC, confusion matrices, and visual diagnostic plots.

## Limitations

These projects were developed as coursework exercises and are intended as applied machine learning demonstrations rather than production-ready systems.

The original datasets are not included if they are course-provided or not publicly shareable. To run the scripts, place the required CSV files in the `data/` folder using the filenames listed above.
