# This catboost classification model the famous titanic dataset. 
# The SHAP package is used for feature importance analysis

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

import catboost
print('catboost version:', catboost.__version__)
from catboost import CatBoostClassifier 

titanic_df = pd.read_csv('https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv')
print(titanic_df.info())

# =====================================
# Claning the data
# =====================================

# strip first letter from cabin number if there; because letters inform us about the floors
titanic_df['cabin'] = titanic_df['cabin'].replace(np.NaN, 'U')      # replacing empty with letter U
titanic_df['cabin'] = [ln[0] for ln in titanic_df['cabin']]         # list comprehension to get the first index
titanic_df['cabin'] = titanic_df['cabin'].replace('U', 'Unknown') 

# Create an isfemale column using 1s and 0s (1 is female)
titanic_df['isfemale'] = np.where(titanic_df['sex'] == 'female', 1, 0)

# drop features not needed for model (excluding columns)
titanic_df = titanic_df[[f for f in list(titanic_df) if f not in ['sex', 'name', 'boat','body', 'ticket', 'home.dest']]]

# make pclass actual categorical column
titanic_df['pclass'] = np.where(titanic_df['pclass'] == 1, 'First', 
                                np.where(titanic_df['pclass'] == 2, 'Second', 'Third'))

# Removing NaN with Unknown
titanic_df['embarked'] = titanic_df['embarked'].replace(np.NaN, 'Unknown') 

# impute age to mean
titanic_df['age'] = titanic_df['age'].fillna(titanic_df['age'].mean())
titanic_df['age']

# This is enough for Catboost, it can handle features like that without the need of encoding 

# =====================================
# mapping categorical values
# =====================================

titanic_catboost_ready_df = titanic_df.dropna() # dropping if empty 

# everything that is not our target variable is defined as a feature
features = [feat for feat in list(titanic_catboost_ready_df) 
            if feat != 'survived']
print(features)
# ['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked', 'isfemale']

# Categorical features are these that are not a float
categorical_features = np.where(titanic_catboost_ready_df[features].dtypes != float)[0]

# =====================================
# Split data into train and test
# =====================================

X_train, X_test, y_train, y_test = train_test_split(titanic_df[features], 
                                                    titanic_df[['survived']],  # looking for survivals (1s)
                                                    test_size=0.3,             # 30%
                                                    random_state=1)

# =====================================
# Defining parameters for catboost
# =====================================

params = {'iterations':2057,
        'learning_rate':0.01,
        'cat_features':categorical_features,
        'depth':3,
        'eval_metric':'AUC', #it's classification so we are looking for Area Under the Curve
        'verbose':200,
        'od_type':"Iter", # overfit detector
        'od_wait':500, # most recent best iteration to wait before stopping
        'random_seed': 1}

# =====================================
# Building catboost model (classification)
# =====================================

cat_model = CatBoostClassifier(**params)
cat_model.fit(X_train, y_train,   
          eval_set=(X_test, y_test), 
          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
          plot=True)

import shap  # package used to calculate Shap values
# SHapley Additive exPlanations
# The goal of SHAP is to explain the prediction of an instance x by computing the 
# contribution of each feature to the prediction. The SHAP explanation method computes 
# Shapley values from coalitional game theory. The feature values of a data instance act as 
# players in a coalition. Shapley values tell us how to fairly distribute the 
# "payout" (= the prediction) among the features. 


from catboost import CatBoostClassifier, Pool
shap_values = cat_model.get_feature_importance(Pool(X_test, label=y_test,cat_features=categorical_features) ,
                                               type="ShapValues")
 
expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]

shap.initjs()
shap.force_plot(expected_value, shap_values[0,:], X_test.iloc[0,:])

print(shap.summary_plot(shap_values, X_test))


# print(titanic_df.info())