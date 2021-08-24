---
title: ML lab - Home data
date: 2021-08-11 11:11:11 -0400
categories: [1CodeNote, MLNote]
tags: [ML]
toc: true
---

- [ML lab - Home data](#ml-lab---home-data)
  - [Step 1: Evaluate several models](#step-1-evaluate-several-models)
  - [Step 2: Generate test predictions](#step-2-generate-test-predictions)


- ref
  - https://www.kaggle.com/learn/intermediate-machine-learning


---


# ML lab - Home data

work with data from the Housing Prices Competition for Kaggle Learn Users to predict home prices in Iowa using 79 explanatory variables describing (almost) every aspect of the homes.



```py
# =============================== Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")  
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")



# =============================== setup ML moduel
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_data_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y_data = X_data_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X_data = X_data_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X_data, y_data, train_size=0.8, test_size=0.2, random_state=0)




X_train.head()
#      LotArea	YearBuilt	1stFlrSF	2ndFlrSF	FullBath	BedroomAbvGr	TotRmsAbvGrd
# Id							
# 619	11694	2007	1828	0	2	3	9
# 871	6600	1962	894	0	1	2	5
# 93	13360	1921	964	0	1	2	5
# 818	13265	2002	1689	0	2	3	7
# 303	13704	2001	1541	0	2	3	6




# =============================== defines five different random forest models
from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, random_state=0, criterion='mae')
model_4 = RandomForestRegressor(n_estimators=200, random_state=0, min_samples_split=20)
model_5 = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=7)

models = [model_1, model_2, model_3, model_4, model_5]





# =============================== select the best model out of the five
# define a function score_model(), returns the mean absolute error (MAE) from the validation set. 
# the best model will obtain the lowest MAE. 
from sklearn.metrics import mean_absolute_error
​
# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)
​
for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
# Model 1 MAE: 24015
# Model 2 MAE: 23740
# Model 3 MAE: 23528
# Model 4 MAE: 23996
# Model 5 MAE: 23706


```



## Step 1: Evaluate several models 

```py
# Fill in the best model
best_model = model_3
​
```


## Step 2: Generate test predictions
 

```py
# Define a model
my_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)



```
















.