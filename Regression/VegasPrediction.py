import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Regression.CustomClasses import import_vegas_data, ReplaceTextBools, DropUselessCols, ReplaceOneHot, print_cor
from Regression.CustomClasses import separate_input_output, GetHotelStars
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#-----------------------------------------------------------
# This script is a play ground for me to learn to use
# sklearn on real data. I realized late in the exercise
# that the data is atrocious for machine learning and there
# isnt much that can be done to make good predictions but
# still continued to explore it !
#-----------------------------------------------------------
vegas = import_vegas_data()

## Test splits
train_set, test_set = train_test_split(vegas,test_size=0.2,random_state=666)
vegas_train = train_set.reset_index(drop=True).copy()
vegas_test = test_set.reset_index(drop=True).copy()

# Get the outcome and the income
vegas_train_input, vegas_train_output = separate_input_output(vegas_train)
vegas_test_input, vegas_test_output = separate_input_output(vegas_test)

# List the different type of columns for the pipeline
all_cats = list(vegas_train_input.columns)
all_cats.remove('Hotel stars')
all_cats.remove('Nr. rooms')
numbers_cats = ['Hotel stars', 'Nr. rooms']

# instantiating the pipeline for the categorical and binary columns
bool_pipeline = Pipeline([
    ('bool_replacer',ReplaceTextBools()),
     ('Drop_useless',DropUselessCols()),
     ('Replace_OneHot',OneHotEncoder()),
])

# instantiating the pipeline for the numerical columns
not_bool_pipeline = Pipeline([
    ('Transform_hotel_start',GetHotelStars()),
    ('min_max', MinMaxScaler()),
])

# Instantiating the main pipeline joining the tow others
main_pipeline = ColumnTransformer([
    ('all', bool_pipeline, all_cats),
    ('not_bool', not_bool_pipeline, numbers_cats),
])

# Passing the inputs in the main pipeline
vegas_train_input = main_pipeline.fit_transform(vegas_train_input)
vegas_test_input = main_pipeline.transform(vegas_test_input)

#Trying linear regression
lr = LinearRegression()
lr.fit(vegas_train_input,vegas_train_output)
predictions = lr.predict(vegas_test_input)
linear_mse = mean_squared_error(predictions,vegas_test_output)
print('Error with linear regression : ', np.sqrt(linear_mse))

# Trying with dt
dt = DecisionTreeRegressor()
dt.fit(vegas_train_input, vegas_train_output)
predictions = dt.predict(vegas_test_input)
DecisionTree_mse = mean_squared_error(predictions,vegas_test_output)
print('Error Decision Tree : ', np.sqrt(DecisionTree_mse))

# TESTING ON CROSS VALIDATION SETS
scores = cross_val_score(dt, vegas_train_input, vegas_train_output, scoring='neg_mean_squared_error', cv=8)
dt_scores = np.sqrt(-scores)

scores = cross_val_score(lr, vegas_train_input, vegas_train_output, scoring='neg_mean_squared_error', cv=8)
lr_scores = np.sqrt(-scores)

#  GRID SEARCH FOR RANDOM FOREST REGRESSOR
parameters_grid = [{'n_estimators':[5,  10, 20], 'max_features':[10,15,20]}]

forrest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forrest_reg,parameters_grid, scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(vegas_test_input,vegas_test_output)

final_model = grid_search.best_estimator_
scores = cross_val_score(final_model, vegas_train_input, vegas_train_output, scoring='neg_mean_squared_error', cv=8)
final_scores = np.sqrt(-scores)

# we can conclude that linear regression still does better than a fine tuned random forest.
# the goal of this not to din the best results possible but to learn to use sklearn, we can say goal accomplished
