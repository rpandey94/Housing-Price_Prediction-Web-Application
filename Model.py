import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# fetching data for modelling
file_path = os.path.join("datasets", "housing")

def fetch_data(file_path=file_path):
    file_path = os.path.join(file_path, "housing.csv")
    return pd.read_csv(file_path)

df = fetch_data()
# print(df.head())

# Split using Stratified Sampling

def strata_samp_split(data, ratio, category):
    strata = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
    for train_index, test_index in strata.split(data, data[category]):
        strata_train_set = data.loc[train_index]
        strata_test_set = data.loc[test_index]
    return strata_train_set, strata_test_set

""" 1. median_income has a high correlation with the target variable median_housing_price
    you can check it with df.corr() function.

    2. Create sampling by creating income categories, this ensures the training and 
    test set have equal proportion of records based on income categories
"""

df_strata = df
df_strata["income_category"] = pd.cut(df_strata["median_income"], 
                                      bins = [0.,1.5,3.,4.5,6., np.inf], labels = [1,2,3,4,5])

train, test = strata_samp_split(df_strata, 0.2, "income_category")

for set_ in (train, test):
    set_.drop("income_category", axis=1, inplace=True)

# Seperating labels from training set
housing_labels = train["median_house_value"].copy()
housing = train.drop("median_house_value", axis=1)

# Creating a transformer to create and add new combination of columns
# New column combinations are made as they may have a stronger correlation with the target variable
# Making transformers will allow us to make a pipeline which can automate the entire process for us
# Each transformer class we make can be sequenced into a pipeline

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class combine_feature(BaseEstimator, TransformerMixin):
    def __init__(self, add_berooms_per_room=False):
        self.add_berooms_per_room = add_berooms_per_room
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_berooms_per_room:
            berooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, 
                         rooms_per_household,
                         population_per_household,
                         berooms_per_room]
        else:
            return np.c_[X, 
                  rooms_per_household, 
                  population_per_household]



# Making a pipeline for tranforming numerical columns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([("imputer", SimpleImputer(strategy = "median")),
                          ("combine_columns", combine_feature()),
                          ("standard", StandardScaler())
                        ])

# Imputer takes numerical dataframe as input only, so make copy with numerical columns only!
# An imputer is used to fill the missing values in any numerical column accross the dataframe with an imputing strategy

housing_num = housing.drop("ocean_proximity", axis=1)

# Creating full pipeline to process categorical and numerical Data together

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Getting column names of numerical attributes
num_attributes = list(housing_num)

#Getting column names of categorical attributes
cat_attributes = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attributes),
                                   ("cat", OneHotEncoder(), cat_attributes)])

# This is our numpy array that we will feed to a machine learning algorithm
housing_prepared = full_pipeline.fit_transform(housing)


# Adding new column names
col = list(housing_num.columns) + ["rooms_per_household", "population_per_household", "<1H OCEAN",
                                   "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

# Generating Dataframe from array to view the prepared dataframe
housing_prepared_df = pd.DataFrame(housing_prepared, columns=col, index=housing.index)
# print(housing_prepared_df.head())  

# Using a RandomForest Predictor
# Implementing grid search to find the best hyper-parameters for our model
# We pass a list of values with their parameter names in a parameter grid
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Total combination of hyper-parameters is 30 + 6 = 36
param_grid = [
              {'n_estimators': [3, 10, 30, 45, 50], 'max_features': [2, 4, 6, 8, 10, 12]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
             ] 

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid,
                           scoring = "neg_mean_squared_error",
                           cv = 5,
                           return_train_score = True)

grid_search.fit(housing_prepared, housing_labels)

# Printing the best set of hyper-parameters after grid search

# grid_search.best_params_

# Final model with the parameter values with best fit to the data

final_model = grid_search.best_estimator_

from sklearn.metrics import mean_squared_error

X_test = test.drop("median_house_value", axis=1)
y_test = test["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

# A way to save scikit-learn models using joblib

import joblib
import os

my_model = final_model
os.makedirs("Saved_Models", exist_ok = True)
path = os.path.join("Saved_Models", "my_model.pkl")
joblib.dump(my_model, path)