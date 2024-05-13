# -*- coding: utf-8 -*-
"""142FinalProj.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PA4PRe7lO3QNrrvCPtyDmMsPjs9p4iJ8

The datasets we used in GoogleSheets form:

1. https://drive.google.com/file/d/1ZtrIWOztk4NI4Y0n9pE82UEBxDTG9AJS/view?usp=drive_link (oasis_cross-sectional.csv)
2. https://drive.google.com/file/d/1s0A6vc3Pkf6gaiusWYtFTsBVw9EoPAg5/view?usp=drive_link (oasis_longitudinal.csv)
3. https://drive.google.com/file/d/1J6S_-FP5oqrVNSPESUEP2Xu1tlWW-g6l/view?usp=drive_link (Alzheimer_s_Disease_and_Healthy_Aging_Data.csv)
4. https://drive.google.com/file/d/1rYh3GIJHCOD3w5uN_bHA3wqTq-vnL5iv/view?usp=drive_link (alzheimer.csv)

You will have to download these csv files, name them exactly as listed in the parantheses, and upload them to your Google Drive (in 'My Drive').
"""

#import dependencies/libraries
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from google.colab import drive
drive.mount('/content/drive')
# Path to your files in Google Drive
file_path1 = '/content/drive/My Drive/oasis_cross-sectional.csv'
file_path2 = '/content/drive/My Drive/alzheimer.csv'


# Reading the files into pandas DataFrame
cross_data = pd.read_csv(file_path1)
aging_data = pd.read_csv(file_path2)

# Modify education columns to support merging

## Rename columns to the same name
aging_data = aging_data.rename(columns ={"EDUC":"Educ"})

## Convert years of education from oasis...csv to 1-5 scale from alzheimer.csv
# 1: less than high school grad., 2: high school grad., 3: some college, 4: college grad., 5: beyond college
def scale(years_of_educ):
  if years_of_educ < 13:
    return 1
  elif years_of_educ == 13:
    return 2
  elif years_of_educ < 17:
    return 3
  elif years_of_educ == 17:
    return 4
  else:
    return 5
aging_data["Educ"] = aging_data["Educ"].apply(scale)

# Check the column names for both DataFrames
common_columns = list(set(cross_data.columns) & set(aging_data.columns))

# Merge with the common columns
combined_data = pd.merge(cross_data, aging_data, how='outer', on=common_columns)
combined_data = combined_data.drop(columns=['Delay', 'MMSE','Group', 'ID', 'Hand'])

# some more discovered correlations in our report?

combined_data['M/F'].replace(['M', 'F'],[1, 0], inplace=True)

combined_data = combined_data[combined_data['CDR'].notna()]

# Fill missing values in SES with -1
if 'SES' in combined_data.columns:
    combined_data['SES'] = combined_data['SES'].fillna(-1)

# Save the combined DataFrame to a new CSV file
output_path = '/content/drive/My Drive/combined_data.csv'

combined_data.to_csv(output_path, index=False)


# Calculate the overlap percentage relative to each dataset
overlapping_data = pd.merge(cross_data, aging_data,  how='inner', on=common_columns)

overlap_percentage_df1 = (len(overlapping_data) / len(cross_data)) * 100
overlap_percentage_df2 = (len(overlapping_data) / len(aging_data)) * 100

# Count of rows with null values for each column
print(pd.isnull(combined_data).sum())

print(overlap_percentage_df1)
print(overlap_percentage_df2)
print(combined_data.head(20))
print(combined_data.shape[0])

# M/F Hand(N/A)  Age  Educ  SES(N/A)  CDR  eTIV   nWBV    ASF
print(combined_data['SES'].value_counts())

#Should we delete the SES N/A rows? (38)

#common columns between datasets
print(common_columns)

#Correlation between CDR and other variables
correlation_matrix = combined_data.corr()

cdr_correlations = correlation_matrix['CDR']

cdr_correlations = cdr_correlations[cdr_correlations.index != 'CDR']

print("Correlations of variables with CDR:")
print(cdr_correlations)

positive_correlations = cdr_correlations[cdr_correlations > 0]
print("\nPositive correlations with CDR:")
print(positive_correlations)

#get correlation graphs of data set
import seaborn as sns

sns.pairplot(combined_data.iloc[:, 1:10], diag_kind='kde')

#split the data into train/test (20%/80%) through random selection

# Creating binary column for CDR
combined_data['CDR_binary'] = np.where(combined_data['CDR'] > 0, 1, 0)

# CDR > 0 -> dementia

x = combined_data.drop(columns=['CDR', 'CDR_binary'])
y = combined_data['CDR_binary']

testsize = np.rint((0.8) * combined_data.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# Baseline Model
print(y_test.value_counts()) # More values are 0 compared to 1

baseline_tn = sum(y_test == 0) # Baseline model predicts 0 for every observation
baseline_tp = 0
baseline_fp = 0
baseline_fn = len(y_test) - baseline_tn

baseline_acc = baseline_tn / len(y_test)
print('Baseline Accuracy: ', baseline_acc)

# CART with CV

grid_values = {'ccp_alpha': np.linspace(0, 0.1, 201), 'random_state': [2023]}

cv = KFold(n_splits=5, random_state=23, shuffle=True)
dtc = DecisionTreeClassifier()

dtc_cv = GridSearchCV(dtc, param_grid = grid_values, scoring = 'accuracy', cv = cv, verbose = 1)

dtc_cv.fit(x_train, y_train)

y_pred = dtc_cv.best_estimator_.predict(x_test)

model_alphavalue = dtc_cv.best_params_['ccp_alpha']

print("Best ccp_alpha value: ", model_alphavalue)

model_acc = accuracy_score(y_test, y_pred)

print('CART Accuracy: ', model_acc)

# Random Forest (with CV on max_features)

grid_values = {'max_features': np.linspace(1,7,7, dtype='int32'),
               'min_samples_leaf': [5],
               'n_estimators': [2000],
               'random_state': [23]}

rf = RandomForestClassifier()
cv = KFold(n_splits=5,random_state=23,shuffle=True)
rf_cv = GridSearchCV(rf, param_grid=grid_values, scoring='accuracy', cv = cv, verbose = 1)
rf_cv.fit(x_train, y_train)

# show parameters selected by CV, max_features = 5
print(rf_cv.best_params_)

# show feature importance
plt.figure(figsize=(8,7))
plt.barh(x_train.columns, 100*rf_cv.best_estimator_.feature_importances_)
plt.show()

# prediction and accuracy
rf_cv_y_pred = rf_cv.best_estimator_.predict(x_test)
rf_acc = accuracy_score(y_test, rf_cv_y_pred)
print('RF Accuracy: ', rf_acc) # surprisingly performs worse than CV CART

# Gradient Boosting with CV (takes ~10 minutes to run)

grid_values = {'n_estimators': np.linspace(100, 2000, 10, dtype='int32'),
               'learning_rate': [0.01],
               'max_leaf_nodes': np.linspace(2, 10, 8, dtype='int32'),
               'max_depth': [100],
               'min_samples_leaf': [10],
               'random_state': [23]}

gbc = GradientBoostingClassifier()
cv = KFold(n_splits=5,random_state=23,shuffle=True)
gbc_cv = GridSearchCV(gbc, param_grid=grid_values, cv=5, verbose=1)
gbc_cv.fit(x_train, y_train)

# Predictions and Accuracy
y_pred_gbc = gbc_cv.predict(x_test) # by default uses best_estimator for predictions
gbc_cv_acc = accuracy_score(y_test, y_pred_gbc)
print('Gradient Boosting Accuracy: ', gbc_cv_acc)

# Gradient Boosting n_estimators vs accuracy plot
print(gbc_cv.best_params_)

n_estimators = gbc_cv.cv_results_['param_n_estimators'].data
gbc_cv_acc_scores = gbc_cv.cv_results_['mean_test_score']

plt.figure(figsize=(12, 8))
plt.xlabel('n_estimators', fontsize=16)
plt.ylabel('GBC CV Accuracy', fontsize=16)
plt.grid(True, which='both')

N = len(grid_values['max_leaf_nodes'])
M = len(grid_values['n_estimators'])
for i in range(N):
    plt.scatter(n_estimators[(M*i):(M*i)+M], gbc_cv_acc_scores[(M*i):(M*i)+M], s=30)
    plt.plot(n_estimators[(M*i):(M*i)+M], gbc_cv_acc_scores[(M*i):(M*i)+M], linewidth=2,
             label='max leaf nodes = '+str(grid_values['max_leaf_nodes'][i]))
plt.legend(loc='lower right')
plt.show()

# Logistic Regression

logreg = LogisticRegression(random_state=23, solver='lbfgs', max_iter=1000)
logreg.fit(x_train, y_train)

y_prob_logreg = logreg.predict_proba(x_test)
y_pred_logreg = pd.Series([1 if x > 0.7 else 0 for x in y_prob_logreg[:,1]], index=y_test.index)

logreg_acc = accuracy_score(y_test, y_pred_logreg)
print('Logistic Regression Accuracy: ', logreg_acc)

# Compare Accuracy, TPR, FPR, PRE values from all models

cm_dtc_cv = confusion_matrix(y_test, y_pred)
cm_rf_cv = confusion_matrix(y_test, rf_cv_y_pred)
cm_gbc_cv = confusion_matrix(y_test, y_pred_gbc)
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

baseline_TPR = baseline_tp / (baseline_tp + baseline_fn)
baseline_FPR = baseline_fp / (baseline_fp + baseline_tn)
baseline_PRE = baseline_tn / (baseline_tn + baseline_fn)

dtc_cv_TPR = cm_dtc_cv.ravel()[3] / (cm_dtc_cv.ravel()[3] + cm_dtc_cv.ravel()[2])
dtc_cv_FPR = cm_dtc_cv.ravel()[1] / (cm_dtc_cv.ravel()[1] + cm_dtc_cv.ravel()[0])
dtc_cv_PRE = cm_dtc_cv.ravel()[3] / (cm_dtc_cv.ravel()[3] + cm_dtc_cv.ravel()[1])

rf_cv_TPR = cm_rf_cv.ravel()[3] / (cm_rf_cv.ravel()[3] + cm_rf_cv.ravel()[2])
rf_cv_FPR = cm_rf_cv.ravel()[1] / (cm_rf_cv.ravel()[1] + cm_rf_cv.ravel()[0])
rf_cv_PRE = cm_rf_cv.ravel()[3] / (cm_rf_cv.ravel()[3] + cm_rf_cv.ravel()[1])

gbc_cv_TPR = cm_gbc_cv.ravel()[3] / (cm_gbc_cv.ravel()[3] + cm_gbc_cv.ravel()[2])
gbc_cv_FPR = cm_gbc_cv.ravel()[1] / (cm_gbc_cv.ravel()[1] + cm_gbc_cv.ravel()[0])
gbc_cv_PRE = cm_gbc_cv.ravel()[3] / (cm_gbc_cv.ravel()[3] + cm_gbc_cv.ravel()[1])

logreg_TPR = cm_logreg.ravel()[3] / (cm_logreg.ravel()[3] + cm_logreg.ravel()[2])
logreg_FPR = cm_logreg.ravel()[1] / (cm_logreg.ravel()[1] + cm_logreg.ravel()[0])
logreg_PRE = cm_logreg.ravel()[3] / (cm_logreg.ravel()[3] + cm_logreg.ravel()[1])

comparison_data = {'Baseline':[baseline_acc,baseline_TPR,baseline_FPR, baseline_PRE],
                   'Decision Tree Classifier w/ CV (CART)':[model_acc,dtc_cv_TPR,dtc_cv_FPR,dtc_cv_PRE],
                   'Random Forest w/ CV':[rf_acc,rf_cv_TPR, rf_cv_FPR,rf_cv_PRE],
                   'Gradient Boosting w/ CV':[gbc_cv_acc,gbc_cv_TPR,gbc_cv_FPR,gbc_cv_PRE],
                   'Logistic Regression':[logreg_acc,logreg_TPR, logreg_FPR,logreg_PRE]}


comparison_table = pd.DataFrame(data=comparison_data, index=['Accuracy', 'TPR', 'FPR','PRE']).transpose()
comparison_table.style.set_properties(**{'font-size': '12pt',}).set_table_styles([{'selector': 'th', 'props': [('font-size', '10pt')]}])
comparison_table

# Bootstrap best model - Gradient Boosting w/ CV
def bootstrap_validation(test_data, test_label, train_label, model, metrics_list, sample=500, random_state=23):
    n_sample = sample
    n_metrics = len(metrics_list)
    output_array=np.zeros([n_sample, n_metrics])
    output_array[:]=np.nan
    print(output_array.shape)
    for bs_iter in range(n_sample):
        bs_index = np.random.choice(test_data.index, len(test_data.index), replace=True)
        bs_data = test_data.loc[bs_index]
        bs_label = test_label.loc[bs_index]
        bs_predicted = model.predict(bs_data)
        for metrics_iter in range(n_metrics):
            metrics = metrics_list[metrics_iter]
            output_array[bs_iter, metrics_iter]=metrics(bs_predicted,bs_label)
    output_df = pd.DataFrame(output_array)
    return output_df

def model_accuracy(y_pred_gbc, y_test):
    return accuracy_score(y_test, y_pred_gbc)

def model_TPR(y_pred_gbc, y_test):
    cm_gbc_cv = confusion_matrix(y_test, y_pred_gbc)
    return cm_gbc_cv.ravel()[3] / (cm_gbc_cv.ravel()[3] + cm_gbc_cv.ravel()[2])

def model_FPR(y_pred_gbc, y_test):
    cm_gbc_cv = confusion_matrix(y_test, y_pred_gbc)
    return cm_gbc_cv.ravel()[1] / (cm_gbc_cv.ravel()[1] + cm_gbc_cv.ravel()[0])

def model_PRE(y_pred_gbc, y_test):
    cm_gbc_cv = confusion_matrix(y_test, y_pred_gbc)
    return cm_gbc_cv.ravel()[3] / (cm_gbc_cv.ravel()[3] + cm_gbc_cv.ravel()[1])

bs_output = bootstrap_validation(x_test,y_test,y_train,gbc_cv,
                                 metrics_list=[model_accuracy, model_TPR, model_FPR, model_PRE],
                                 sample = 5000)

# Bootstrap estimate graphs to analyze variability in accuracy, TPR, FPR, PRE values
fig, axs = plt.subplots(ncols=4, figsize=(18,6))
print('Mean Values for Each Metric: ', [np.mean(bs_output.iloc[:,i]) for i in range(len(bs_output.columns))])

axs[0].set_xlabel('GBC CV Bootstrap ACC Estimate', fontsize=10)
axs[0].set_ylabel('Count', fontsize=16)
axs[0].hist(bs_output.iloc[:,0], bins=20,edgecolor='green', linewidth=2,color = "grey")
axs[0].axvline(x=np.mean(bs_output.iloc[:,0]), color='red', linestyle='--')

axs[1].set_xlabel('GBC CV Bootstrap TPR Estimate', fontsize=10)
axs[1].hist(bs_output.iloc[:,1], bins=20,edgecolor='green', linewidth=2,color = "grey")
axs[1].axvline(x=np.mean(bs_output.iloc[:,1]), color='red', linestyle='--')

axs[2].set_xlabel('GBC CV Bootstrap FPR Estimate', fontsize=10)
axs[2].hist(bs_output.iloc[:,2], bins=20,edgecolor='green', linewidth=2,color = "grey")
axs[2].axvline(x=np.mean(bs_output.iloc[:,2]), color='red', linestyle='--')

axs[3].set_xlabel('GBC CV Bootstrap PRE Estimate', fontsize=10)
axs[3].hist(bs_output.iloc[:,3], bins=20,edgecolor='green', linewidth=2,color = "grey")
axs[3].axvline(x=np.mean(bs_output.iloc[:,3]), color='red', linestyle='--')

# Simple CART Model

# Simple CART 1 - enforcing max_depth

simple_dtc = DecisionTreeClassifier(min_samples_leaf = 5,
                                    ccp_alpha = 0.002,
                                    max_depth = 3
                                    random_state = 88)

simple_dtc = simple_dtc.fit(x_train, y_train)
simple_dtc_y_pred = simple_dtc.predict(x_test)
simple_dtc_acc = accuracy_score(y_test, simple_dtc_y_pred)
print("Accuracy for this model is", simple_dtc_acc)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

print('Node count =', simple_dtc.tree_.node_count)
plt.figure(figsize=(12,12))
plot_tree(simple_dtc,
          feature_names=x_train.columns,
          class_names=['0','1'],
          filled=True,
          impurity=True,
          rounded=True,
          fontsize=12)
plt.show()

# Simple CART 2 - changing min_samples_leaf


simple_dtc = DecisionTreeClassifier(min_samples_leaf=50,
                                    ccp_alpha=0.002,
                                    random_state = 88)

simple_dtc = simple_dtc.fit(x_train, y_train)
simple_dtc_y_pred = simple_dtc.predict(x_test)
simple_dtc_acc = accuracy_score(y_test, simple_dtc_y_pred)
print("Accuracy for this model is", simple_dtc_acc)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

print('Node count =', simple_dtc.tree_.node_count)
plt.figure(figsize=(12,12))
plot_tree(simple_dtc,
          feature_names=x_train.columns,
          class_names=['0','1'],
          filled=True,
          impurity=True,
          rounded=True,
          fontsize=12)
plt.show()

# Simple CART 3 - changing ccp_alpha to 0.1


simple_dtc = DecisionTreeClassifier(min_samples_leaf=5,
                                    ccp_alpha=0.1,
                                    random_state = 88)

simple_dtc = simple_dtc.fit(x_train, y_train)
simple_dtc_y_pred = simple_dtc.predict(x_test)
simple_dtc_acc = accuracy_score(y_test, simple_dtc_y_pred)
print("Accuracy for this model is", simple_dtc_acc)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

print('Node count =', simple_dtc.tree_.node_count)
plt.figure(figsize=(12,12))
plot_tree(simple_dtc,
          feature_names=x_train.columns,
          class_names=['0','1'],
          filled=True,
          impurity=True,
          rounded=True,
          fontsize=12)
plt.show()
