#!/usr/bin/env python
# coding: utf-8

# Here are some of the resources that were used for the assignment:

# https://realpython.com/linear-regression-in-python/
# 
# https://www.datacamp.com/tutorial/save-as-csv-pandas-dataframe
# 
# https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# 
# https://www.youtube.com/watch?v=Liv6eeb1VfE
# 
# https://data36.com/polynomial-regression-python-scikit-learn/
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
# 
# https://numpy.org/doc/2.2/user/basics.indexing.html
# 
# https://matplotlib.org/stable/users/explain/quick_start.html#types-of-inputs-to-plotting-functions
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
# 
# https://matplotlib.org/stable/users/explain/figure/figure_intro.html
# 
# https://stackoverflow.com/questions/33711985/flattening-a-list-of-numpy-arrays
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
# 
# https://numpy.org/doc/2.2/reference/generated/numpy.argmax.html#numpy-argmax
# 
# https://numpy.org/doc/2.2/reference/generated/numpy.char.chararray.flatten.html#numpy.char.chararray.flatten
# 
# https://stackoverflow.com/questions/11549486/linearregression-returns-list-within-list-sklearn
# 
# https://www.youtube.com/watch?v=gJo0uNL-5Qw
# 
# https://www.youtube.com/watch?v=IhSWvwmpwTU
# 

# -----------------------------------------------------------------------------------------------------------------------

# A.1.1) Data Loading and Basic Plot: Load the dataset, assign the correct columns as features (X) and target (y). Create a scatter plot showing the relationship between a shell measurement (e.g. length) and Rings. Label the axes (Shell Length (mm) vs. Number of Rings) and add a suitable title.

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/maxib/Downloads/turtle-2.csv")

plt.figure(figsize=(10, 15))
plt.scatter(data['Length'], data['Rings'], color = "green")
plt.xlabel('Shell Length (mm)')
plt.ylabel('Number of Rings')
plt.title('Relationship between shell length and number of rings in turtles')
plt.show()


# (A.1.2) Summary Statistics and Correlations: Calculate basic descriptive
# statistics (mean, std, min, max) for numerical features. Generate a correlation heatmap to identify features most correlated with Rings. State
# which feature correlates most strongly with Rings.

# In[6]:


data_numeric = data.select_dtypes(include=np.number)
data_mean = data_numeric.mean()
data_std = data_numeric.std()
print("Mean:", data_mean)
print("Standard Deviation:", data_std)

data_corr = data_numeric.corr()
print("Correlation Matrix:\n", data_corr)

plt.figure(figsize=(7, 5))
sns.heatmap(data_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.8)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

max_feature, max_value = data_corr['Rings'].drop('Rings').idxmax(), data_corr['Rings'].drop('Rings').max()
min_feature, min_value = data_corr['Rings'].drop('Rings').idxmin(), data_corr['Rings'].drop('Rings').min()
print(f'Most correlated feature: {max_feature} with correlation value: {max_value}')
print(f'Least correlated feature: {min_feature} with correlation value: {min_value}')


# (A.1.3) Categorical Influence (Gender): For the gender column, visualize
# the average rings (age) per gender category (e.g. box plot or violin plot).
# Discuss whether gender appears to have a noticeable influence on the
# turtle’s age.

# In[7]:


plt.figure(figsize=(7, 5))
sns.boxplot(x='Sex', y='Rings', data=data, palette='muted')
plt.xlabel('Gender')
plt.ylabel('Number of Rings (Age)')
plt.title('Influence of Gender on Turtle Age (Number of Rings)')
plt.show()


# In[8]:


# Discussion: gender appears to have influence on age, mainly due to the I (immature) category. The male and female turtles have the almost
# the same median while the I category has lower median. From the box we can conclude as well that the male and female turtles have more variability
# in their range compared to the immature turtles, which suggests that gender has greater influence if it is a comparison between I category and one of the
# other two. Otherwise there is not much of difference between Male and Female turtles.


# ----------------------------------------------------------------------------------------------------

# 
# 
# 
# A.2 Regression Tasks
# 
# Split the dataset into 80% training, 10% validation, and 10% testing. Since
# Rings is continuous, bin it before doing a stratified split (e.g. via KBinsDiscretizer
# or np.digitize()) to ensure even distribution of binned target values.

# In[9]:


import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold


# (A.2.1) Train–Val–Test Split: Clearly describe the binning process, then split
# into 80%-10%-10%. Confirm that you preserve the proportion of each bin
# across splits.
# 

# In[10]:


binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
data['Rings_binned'] = binner.fit_transform(data[['Rings']])

train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['Rings_binned'], random_state=32)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['Rings_binned'], random_state=22)

del train_data['Rings_binned'], val_data['Rings_binned'], test_data['Rings_binned']


# (A.2.2) Linear Regression: Train a simple linear regression to predict Rings.
# Evaluate on the test set using R2, MSE, and MAE. Briefly comment on
# the performance. 

# In[31]:


features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
target = 'Rings'

X_train, y_train = train_data[features], train_data[target]
X_val, y_val = val_data[features], val_data[target]
X_test, y_test = test_data[features], test_data[target]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R^2: {r2_score(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")


# In[12]:


# Discussion: The R^2 is not that high, 53 % of the variance in  the dependent variable age can be explained by the model, which  suggests that there are more factors that influence age.
# The MSE and MAE are not large which suggests decent model performance.


# (A.2.3) Polynomial Regression (Degrees 2–5): For polynomial degrees {2, 3,
# 4, 5}, train separate models in a loop. Evaluate each on the validation set
# to pick the best degree, then evaluate that chosen degree on the test set.

# In[34]:


best_degree = None
best_r2 = -np.inf
best_model = None

for degree in range(2, 6):
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X_train, y_train)
    val_pred = poly_model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)

    print(f"Polynomial degree {degree} has validation R^2: {val_r2:.2f}")

    if val_r2 > best_r2:
        best_r2 = val_r2
        best_degree = degree
        best_model = poly_model

print(f"Best polynomial degree: {best_degree}")

test_pred = best_model.predict(X_test)
print(f"Test R^2 (Degree {best_degree}): {r2_score(y_test, test_pred):.2f}")
print(f"Test MSE: {mean_squared_error(y_test, test_pred):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")


# A.2.4) K-Nearest Neighbors (KNN): For k = {1, 3, 6, 10}, train KNN regressors in a loop. Evaluate on the validation set, pick the best k, then
# evaluate that model on the test set.

# In[35]:


best_k = None
best_knn_r2 = -np.inf
best_knn_model = None

for k in [1, 3, 6, 10]: 
    knn_model = KNeighborsRegressor(n_neighbors=k)  
    knn_model.fit(X_train, y_train)  
    val_pred = knn_model.predict(X_val)  
    val_r2 = r2_score(y_val, val_pred)  

    print(f"KNN with k={k} has validation R^2: {val_r2:.2f}")

    if val_r2 > best_knn_r2:  
        best_knn_r2 = val_r2
        best_k = k
        best_knn_model = knn_model

print(f"Best K for KNN: {best_k} with Validation R^2: {best_knn_r2:.2f}")

test_pred = best_knn_model.predict(X_test)

print(f"Test R^2 (K={best_k}): {r2_score(y_test, test_pred):.2f}")
print(f"Test MSE: {mean_squared_error(y_test, test_pred):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")


# (A.2.5) Visual Comparison (Actual vs. Predicted): On the test set, plot
# a scatter plot of actual vs. predicted Rings. Optionally overlay a line or
# curve from the best model to visualize how well it matches the ground
# truth.

# In[15]:


plt.figure(figsize=(10, 8))
plt.scatter(y_test, test_pred, alpha=0.7, edgecolors='brown')
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Actual vs. Predicted Rings")
plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------------

# B.1 Fold-by-Fold Performance & Feature Importance (15
# points)

# Perform a 10-fold cross-validation for the best model (chosen from A.2).

# (B.1.1) Per-Fold Metrics: Collect R2, MSE, and MAE for each of the 10 folds.
# Plot these errors (e.g. line or scatter plot) to see the variation across folds.

# In[28]:


best_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

kf = KFold(n_splits=10, shuffle=True, random_state=27)
r2_scores = []
mse_scores = []
mae_scores = []


for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    best_model.fit(X_fold_train, y_fold_train)
    y_fold_pred = best_model.predict(X_fold_val)

    r2_scores.append(r2_score(y_fold_val, y_fold_pred))
    mse_scores.append(mean_squared_error(y_fold_val, y_fold_pred))
    mae_scores.append(mean_absolute_error(y_fold_val, y_fold_pred))

plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), mse_scores, marker='o', linestyle='-', label="MSE")
plt.plot(range(1, 11), mae_scores, marker='s', linestyle='--', label="MAE")
plt.xlabel("Fold Number")
plt.ylabel("Error")
plt.title("Cross-Validation Error per Fold")
plt.legend()
plt.show()

print(f"Mean R^2: {np.mean(r2_scores):.2f}")
print(f"Mean MSE: {np.mean(mse_scores):.2f}")
print(f"Mean MAE: {np.mean(mae_scores):.2f}")


# (B.1.2) Feature Importance / Coefficients: In the best fold (lowest MSE or
# MAE), investigate which features were most influential. For linear/polynomial models, interpret the coefficients; for tree-based methods or others
# with feature importances , show a bar chart ranking features by importance.

# In[22]:


best_model.fit(X_train, y_train)

coefficients = best_model.named_steps['linearregression'].coef_
feature_names = best_model.named_steps['polynomialfeatures'].get_feature_names_out(input_features=features)

feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)

print("Feature Coefficients:")
print(feature_importance)

plt.figure(figsize=(8, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature coefficients) in best model')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()


# -----------------------------------------------------------------------------------------------------------------------------------------

# B.2 Investigating Errors

# (B.2.1) Worst-Performing Fold: Identify the worst-performing fold in crossvalidation (by MSE or MAE).

# In[ ]:


worst_fold_idx = np.argmax(mse_scores)  
folds = list(kf.split(X_train))
train_index, val_index = folds[worst_fold_idx]
worst_fold_X_val, worst_fold_y_val = X_train.iloc[val_index], y_train.iloc[val_index]
worst_fold_pred = best_model.predict(worst_fold_X_val)



# (B.2.2) Over-Predictions: Within that fold, examine cases where predicted
# Rings > actual Rings. Use a histogram or similar plot of key features
# for these samples. Look for patterns (are these turtles heavier, bigger
# shells, etc.?).

# In[24]:


over_predicted = worst_fold_pred > worst_fold_y_val

plt.figure(figsize=(6, 4))
plt.hist(worst_fold_X_val[over_predicted]['Shell weight'], bins=15, color='lightgreen', edgecolor='black', alpha=0.5)
plt.xlabel("Shell Weight")
plt.ylabel("Count")
plt.title("Distribution of shell weight in over-predictions")
plt.show()


# (B.2.3) Under-Predictions: Do the same for cases where predicted Rings <
# actual. Plot error magnitude vs. a key feature to see whether certain
# types of turtles are systematically under-predicted.

# In[25]:


under_predicted = worst_fold_pred < worst_fold_y_val
error_magnitude = np.abs(worst_fold_pred - worst_fold_y_val)

plt.figure(figsize=(6,4))
plt.scatter(worst_fold_X_val[under_predicted]['Shell weight'], error_magnitude[under_predicted], color='teal', alpha=0.6, edgecolors='black')
plt.xlabel("Shell Weight")
plt.ylabel("Prediction Error Magnitude")
plt.title("Under-Predictions: Error vs. Shell Weight")
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------------------------------

# B.3 Storing & Reporting Results

# (B.3.1) Summary DataFrame: Create a pandas DataFrame summarizing your
# final models: Model Name, Hyperparameters, R2, MSE, MAE.

# In[29]:


summary_df = pd.DataFrame({
    'Model Name': ['Linear Regression', 'Polynomial Regression', 'KNN Regression'],
    'Hyperparameters': ['N/A', f'Degree={best_degree}', f'K={best_k}'],
    'R^2': [r2_score(y_test, model.predict(X_test)),
           r2_score(y_test, best_model.predict(X_test)),
           r2_score(y_test, best_knn_model.predict(X_test))],
    'MSE': [mean_squared_error(y_test, model.predict(X_test)),
            mean_squared_error(y_test, best_model.predict(X_test)),
            mean_squared_error(y_test, best_knn_model.predict(X_test))],
    'MAE': [mean_absolute_error(y_test, model.predict(X_test)),
            mean_absolute_error(y_test, best_model.predict(X_test)),
            mean_absolute_error(y_test, best_knn_model.predict(X_test))]
})

print(summary_df)


# (B.3.2) CSV Output: Save this DataFrame to a CSV file for documentation.
# This ensures results are easily tracked.

# In[19]:


summary_df.to_csv("model_results.csv", index=False)


# In[ ]:




