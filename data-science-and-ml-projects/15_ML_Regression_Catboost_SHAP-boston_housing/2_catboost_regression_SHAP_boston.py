# CatBoostRegressor for predicting house prices in boston area

from sklearn.datasets import load_boston
boston_dataset = load_boston()

# print(boston_dataset.keys())
# print(boston_dataset.DESCR)
# print(boston_dataset.feature_names)
# for ln in boston_dataset.DESCR.split('\n'):
#     print(ln)

import pandas as pd

# Predictors
boston = pd.DataFrame(boston_dataset.data , columns = boston_dataset.feature_names)
print(boston.head(10))

# Target variable
boston['MEDV'] = boston_dataset.target

from catboost import CatBoostRegressor

# data split
outcome_name = 'MEDV'
features_for_model = [f for f in list(boston) if f not in [outcome_name, 'TAX']]

import numpy as np
# get categories and cast to string
boston_categories = np.where([boston[f].apply(float.is_integer).all() for f in features_for_model])[0]
print('boston_categories:', boston_categories)

# convert to values to string
for feature in [list(boston[features_for_model])[f] for f in list(boston_categories)]:
    print(feature)
    boston[feature] = boston[feature].to_string()


# data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston[features_for_model], 
                                                 boston[outcome_name], 
                                                 test_size=0.3, 
                                                 random_state=1)




params = {'iterations':5000,
        'learning_rate':0.001,
        'depth':3,
        'loss_function':'RMSE',
        'eval_metric':'RMSE',
        'random_seed':55,
        'cat_features':boston_categories,
        'metric_period':200,  
        'od_type':"Iter",  
        'od_wait':20,  
        'verbose':True,
        'use_best_model':True}


model_regressor = CatBoostRegressor(**params)

model_regressor.fit(X_train, y_train, 
          eval_set=(X_test, y_test),  
          use_best_model=True,  
          plot= True   
         )

import shap
from catboost import Pool
shap_values = model_regressor.get_feature_importance(Pool(X_test, label=y_test,cat_features=boston_categories) ,
                                               type="ShapValues")
 
expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]

shap.initjs()
shap.force_plot(expected_value, shap_values[0,:], X_test.iloc[0,:])

print(shap.summary_plot(shap_values, X_test))

