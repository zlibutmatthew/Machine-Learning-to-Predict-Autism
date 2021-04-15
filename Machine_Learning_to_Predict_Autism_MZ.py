# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/zlibutmatthew/Machine-Learning-to-Predict-Autism/blob/main/Machine_Learning_to_Predict_Autism_MZ.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="4xUr7djsQIha"
# **Matthew Zlibut**
#
# DATA set description
#
# Autism spectrum disorder (ASD) is a developmental disorder that affects communication and behavior. Unfortunately, waiting for an ASD diagnosis is lengthy and procedures are expensive. The economic impact of autism and the increase in the number of ASD cases across the world reveals an urgent need for the development of easily implemented and effective ASD screening methods.
#
# Column variables presented in this data: **AGE, GENDER, ETHNICITY, JAUNDICE, FAMILY with PDD, TEST TAKER, COUNTRY, etc.**
#
# The **age** of the patient was a number presented in years old. 
#
# **Gender** was only measured to be M or F, which is translated to a 1 or 0 in our data.
#
# **Ethnicity** was a string which lists ethnicities in text format. 
#
# Born with **jaundice** is a Boolean value (True or False)
#
# **Family member with PDD** is a Boolean value (True or False)
#
# **Who is completing the test** is a String value. ex: Parent, self, caregiver, medical staff, clinician ,etc.
#
# **Country of residence** is a String, List countries in text format
#
# **Used the screening app before** Boolean (yes or no) Whether the user has used a screening app
#
# Question 1-10 Answer Binary (0, 1) The answer code of the question based on the screening method used
#
# Screening Score Integer The final score obtained based on the scoring algorithm of the
# screening method used. This was computed in an automated manner
#
#

# + id="xjoTKb3YQIhc" outputId="62df358a-00b5-438f-9b39-dd8fc65adead" colab={"base_uri": "https://localhost:8080/", "height": 258}
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def cross_validation(estimator, param_grid, X_train, y_train):
    #returns a grid search object and cross-validatied scores in that order
    gs = GridSearchCV(estimator=estimator,
                 param_grid = param_grid,
                 scoring='accuracy',
                 cv=5, iid=False, refit=True,
                n_jobs=-1)
    scores = cross_validate(estimator=gs, X=X_train, y=y_train, cv=10, scoring='accuracy', return_train_score=True)
    return gs, scores

def print_scoring_metrics(method, y_true, y_pred, estimator, normalize=False):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(('%s score'+4*'%s %2.2f\t')%(method, 'Accuracy', acc, 'Recall', rec, 'Precision', prec, 'F1', f1))
    

df = pd.read_csv('Autism-Child-Data.csv', na_values = '?')
df = df.dropna(axis=0)

df = pd.get_dummies(df, columns=['country_of_res'], prefix = 'country')
df = pd.get_dummies(df, columns=['ethnicity'])
df = pd.get_dummies(df, columns=['relation'])

df['relation_Self'] += df['relation_self']
df = df.drop(axis=1, columns=['country_Italy',
                              'ethnicity_Others',
                             'relation_Parent',
                             'relation_self',
                             'age_desc',
                             'result'])
df['gender'] = df['gender'].map({'m': 0, 'f': 1})

yn_mapping = {'yes':1,'YES':1,'no':0, 'NO':0}
for label in ['jaundice', 'autism', 'class', 'used_app_before']:
    df[label] = df[label].map(yn_mapping)
    
df.head()
df.tail()

# + id="VyPl2XXdQIhi" outputId="c51329cd-21a3-49f8-863b-ad1d71ac3f81" colab={"base_uri": "https://localhost:8080/", "height": 462}
#SVC Data
clf = SVC(gamma='auto')
X = df.drop(axis=1, columns='class').values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf.fit(X_train_std, y_train) 
clf.predict(X_test_std)
clf.score(X_test_std, y_test, sample_weight=None)

C_range = np.logspace(-2, 4, 10)
gamma_range = np.logspace(-4, 2, 10)

param_grid = [{'C': C_range, 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': gamma_range, 'degree':[2,3,4]}]

gs, scores = cross_validation(clf, param_grid, X_train_std, y_train)
gs.fit(X_train_std, y_train)

print(scores['test_score'].mean())

best_SVC = gs.best_estimator_
print(best_SVC.get_params())
print_scoring_metrics('Optimized SVC', y_test, best_SVC.predict(X_test_std), best_SVC)

# + id="Z58tsWX_QIhl" outputId="22e284d6-051a-4e15-9d52-24c725719d12" colab={"base_uri": "https://localhost:8080/", "height": 496}
#LR
lr = LogisticRegression(C=0.1, solver='lbfgs', multi_class='auto')
lr.fit(X_train_std, y_train)
lr.predict(X_test_std)
lr.score(X_test_std, y_test)

C_range = np.logspace(-2, 4, 10)
l1_range = np.linspace(0.1, 1, 10)

param_grid = [{'C': C_range, 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
             ]#{'C': C_range, 'penalty': ['elasticnet'], 'l1_ratio': l1_range, 'solver': ['saga']}]

gs, scores = cross_validation(lr, param_grid, X_train_std, y_train)
gs.fit(X_train_std, y_train)

print(scores['test_score'].mean())


# + id="WP-0ht5DQIhp" outputId="43bf3ea3-d1dd-49ec-b289-4915a1915a03" colab={"base_uri": "https://localhost:8080/", "height": 71}
best_lr = gs.best_estimator_
print(best_lr.get_params())
print_scoring_metrics('Optimized lr', y_test, best_lr.predict(X_test_std), best_lr)

# + id="FnAIkuk1QIhs" outputId="607af7f9-ac2b-430c-93b6-95931e94dd58" colab={"base_uri": "https://localhost:8080/", "height": 677}
#MLP
mlp = MLPClassifier(solver='lbfgs')
mlp.fit(X_train_std, y_train)
mlp.predict(X_test_std)
mlp.score(X_test_std, y_test)

C_range = np.logspace(-2, 4, 10)
l1_range = np.linspace(0.1, 1, 10)

param_grid = [{'learning_rate': ['constant', 'invscaling', 'adaptive'], 'solver': ['lbfgs', 'sgd', 'adam'],
               'activation':['identity', 'logistic', 'tanh', 'relu'],
               'hidden_layer_sizes': [(73,), (73,37), (73,73,37)]}]

gs, scores = cross_validation(mlp, param_grid, X_train_std, y_train)
gs.fit(X_train_std, y_train)

print(scores)
               
best_mlp = gs.best_estimator_

# + id="AimmbH7ZVngQ" outputId="dcf063bb-7498-468d-85ea-06838f69c5b8" colab={"base_uri": "https://localhost:8080/", "height": 71}
print(best_mlp.get_params())
print_scoring_metrics('Optimized MLP', y_test, best_mlp.predict(X_test_std), best_mlp)
