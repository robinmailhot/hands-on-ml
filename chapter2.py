import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Customs import import_vegas_data, ReplaceTextBools, DropUselessCols, ReplaceOneHot, print_cor
from Customs import separate_input_output, GetHotelStars
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

vegas = import_vegas_data()

## Test splits
train_set, test_set = train_test_split(vegas,test_size=0.2,random_state=666)
vegas_train = train_set.reset_index(drop=True).copy()
vegas_test = test_set.reset_index(drop=True).copy()

# Get the outcome and the income
vegas_train_input, vegas_train_output = separate_input_output(vegas_train)
vegas_test_input, vegas_test_output = separate_input_output(vegas_test)

# List the different type of columns for the pipelin
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
print(np.sqrt(linear_mse))



print('moma')