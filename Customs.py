import os
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator
import os
import xlrd



#path=os.path.join(os.getcwd(),
def import_vegas_data(path='data\\lasvegas_data.csv'):
    """
    path : path to the CSV file
    separation : what the separation in the CSV is
    """
    df = pd.read_csv(path, sep=';' ,header=0)
    return df

class ReplaceTextBools(BaseEstimator, TransformerMixin):
    #this class replaces text bools for real bools (1 and 0)
    def __init__(self,replace=True): # we use it like in sklearn, instantiate, then we will use transform
        self.replace = replace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.replace:
            bool_cats = X.columns[X.nunique()<=2]
            bool_encoder = OrdinalEncoder()
            bool_data = bool_encoder.fit_transform(X[bool_cats]).astype(int)
            X[bool_cats] = bool_data
        return X

class DropUselessCols(BaseEstimator, TransformerMixin):
    #this class drops the useless columns
    def __init__(self,drop=True): # we use it like in sklearn, instantiate, then we will use transform
        self.drop = drop
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.drop:
            to_drop = ['User country','Nr. reviews','Nr. hotel reviews','Helpful votes',
                       'Member years','Review weekday','Review month', 'Hotel name']
            X = X.drop(columns=to_drop,errors='ignore')
        return X

class ReplaceOneHot(BaseEstimator, TransformerMixin):
    #this replaces the column we want with one hot encoded versions
    def __init__(self,replace=True): # we use it like in sklearn, instantiate, then we will use transform
        self.replace = replace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.replace:
            to_one_hot = ['Period of stay','Traveler type','User continent']
            one_hot = OneHotEncoder()
            one_hot_data = one_hot.fit_transform(X[to_one_hot]).astype(int)
            new_names = np.reshape(one_hot.get_feature_names(), (-1, 1))
            X = X.drop(columns=to_one_hot, errors='ignore')
            B = pd.DataFrame(one_hot_data.toarray(),columns=new_names)
            X = pd.concat([X, B], axis=1)
        return X

def print_cor(df):
    correlation_matrix = df.corr()
    print(correlation_matrix['Score'].sort_values(ascending=False))

def separate_input_output(df):
    return df.drop(columns=['Score']) , df['Score'].copy()

class HotelStarsToInt(BaseEstimator, TransformerMixin):
    #this replaces the column we want with one hot encoded versions
    def __init__(self,replace=True):# we use it like in sklearn, instantiate, then we will use transform
        self.replace = replace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.replace:
            X['Hotel stars'] = X['Hotel stars'].astype(np.float32)
        return X

class GetHotelStars(BaseEstimator, TransformerMixin):
    def __init__(self,replace=True):# we use it like in sklearn, instantiate, then we will use transform
        self.replace = replace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.replace:
            for n in range(X['Hotel stars'].shape[0]):
                if len(X['Hotel stars'][n]) == 1:
                    X.loc[n,'Hotel stars'] = float(X.loc[n,'Hotel stars'])

                elif len(X['Hotel stars'][n]) == 3:
                    X.loc[n, 'Hotel stars'] = float(X.loc[n, 'Hotel stars'][0]) + 0.1 * float(X.loc[n,'Hotel stars'][2])
        return X


class MinMaxDrop(BaseEstimator, TransformerMixin):
    def __init__(self,replace=True):# we use it like in sklearn, instantiate, then we will use transform
        self.replace = replace
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.replace:
            X=1
        return X



