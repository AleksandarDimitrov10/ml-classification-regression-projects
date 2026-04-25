#!/usr/bin/env python
# coding: utf-8

# References:

# •	RandomizedSearchCV. (n.d.-c). Scikit-learn.
# https://scikitlearn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
# 
# •	pandas.DataFrame.drop — pandas 2.2.3 documentation. (n.d.-b). https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
# 
# •	pandas.DataFrame.select_dtypes — pandas 2.2.3 documentation. (n.d.). https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes
# 
# •	SimpleImputer. (n.d.). Scikit-learn. 
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# 
# •	OneHotEncoder. (n.d.). Scikit-learn.                                                                    
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# 
# •	ColumnTransformer. (n.d.). Scikit-learn.                                                             
# https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
# 
# •	make_pipeline. (n.d.-b). Scikit-learn. 
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
# 
# •	classification_report. (n.d.). Scikit-learn. ‘
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# 
# •	confusion_matrix. (n.d.). Scikit-learn. 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# 
# •	roc_auc_score. (n.d.). Scikit-learn. 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# 
# •	RocCurveDisplay. (n.d.). Scikit-learn. 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay.from_predictions
# 
# 
# •	PrecisionRecallDisplay. (n.d.). Scikit-learn.
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay.from_predictions
# 
# •	matplotlib.backends.backend_ps — Matplotlib 3.10.1 documentation. (n.d.).
# https://matplotlib.org/stable/api/backend_ps_api.html#matplotlib.backends.backend_ps.RendererPS.set_color
# 
# •	matplotlib.collections—Matplotlib 3.10.1 documentation. (n.d.). https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
# 
# •	LogisticRegression. (n.d.). Scikit-learn. 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# •	1.4. Support vector machines. (n.d.). Scikit-learn. 
# https://scikit-learn.org/stable/modules/svm.html
# 
# •	Krishnaik. (n.d.). GitHub - krishnaik06/GRIDSearchCV. GitHub. https://github.com/krishnaik06/GridSearchCV
# 
# •	Scikit-Learn. (n.d.). GitHub - scikit-learn/scikit-learn: scikit-learn: machine learning in Python. GitHub. https://github.com/scikit-learn/scikit-learn
# 
# •	Michael-Signorotti. (n.d.). scikit-learn-pipeline-example/pipeline_example.py at master · Michael-Signorotti/scikit-learn-pipeline-example. GitHub. 
# https://github.com/Michael-Signorotti/scikit-learn-pipeline-example/blob/master/pipeline_example.py
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Code:

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

df = pd.read_csv("C:/Users/maxib/Downloads/bots_vs_users.csv")  

X = df.drop(columns=['target'])
y = df['target']


num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('target')
cat_cols = df.select_dtypes(include=['object']).columns


numeric_transformer = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='unknown'),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


param_dist_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

rand_search_lr = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_distributions=param_dist_lr,
    n_iter=5,  
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
rand_search_lr.fit(X_train_preprocessed, y_train)

print("Best Logistic Regression Params:", rand_search_lr.best_params_)


param_dist_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

rand_search_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_distributions=param_dist_svm,
    n_iter=5,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
rand_search_svm.fit(X_train_preprocessed, y_train)

print("Best SVM Params:", rand_search_svm.best_params_)

for name, model in [('Logistic Regression', rand_search_lr.best_estimator_), ('SVM', rand_search_svm.best_estimator_)]:
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test_preprocessed)
    y_proba = model.predict_proba(X_test_preprocessed)[:, 1]
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
    roc_display.line_.set_color('green')
    roc_display.line_.set_linestyle('--')
    plt.title(f"{name} ROC Curve")
    plt.grid(True)
    plt.show()

    pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    pr_display.line_.set_color('green')
    pr_display.line_.set_linestyle('--')
    plt.title(f"{name} Precision-Recall Curve")
    plt.grid(True)
    plt.show()


print("\n=== Feature Importance (Top 10) ===")
feature_names = preprocessor.get_feature_names_out()
lr_coef = rand_search_lr.best_estimator_.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_coef
}).sort_values(by='Coefficient', key=abs, ascending=False)
print(feature_importance.head(10))

print("\n=== Gender Group Comparison ===")

gender_test = X_test.copy().reset_index(drop=True)
gender_test['actual'] = y_test.reset_index(drop=True)
gender_test['pred_lr'] = rand_search_lr.best_estimator_.predict(X_test_preprocessed)
gender_test['pred_svm'] = rand_search_svm.best_estimator_.predict(X_test_preprocessed)

print("\nError Rates by Gender (Logistic Regression):")
for g in gender_test['gender'].unique():
    subset = gender_test[gender_test['gender'] == g]
    errors = (subset['pred_lr'] != subset['actual']).mean()
    print(f"Gender {g}: Error rate = {errors:.3f}")


print("\nError Rates by Gender (SVM):")
for g in gender_test['gender'].unique():
    subset = gender_test[gender_test['gender'] == g]
    errors = (subset['pred_svm'] != subset['actual']).mean()
    print(f"Gender {g}: Error rate = {errors:.3f}")


# 

# In[ ]:




