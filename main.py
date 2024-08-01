import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

errors= {}

def load_pre_process_calc_aqi(name):
    df = pd.read_csv(name)

    #cleaning column names 
    df.columns = df.columns.str.strip().astype(str)

    #converting column values to int
    for col in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #creating date object 
    df['date'] = pd.to_datetime(df['date'])

    # sort the dataframe by the 'date' column
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df['month'] = range(len(df))

    pm25_mean = df['pm25'].mean().round(0)
    pm10_mean = df['pm10'].mean().round(0)
    o3_mean = df['o3'].mean().round(0)
    no2_mean = df['no2'].mean().round(0)
    so2_mean = df['so2'].mean().round(0)
    co_mean = df['co'].mean().round(0)

    df['pm25'] = df['pm25'].fillna( pm25_mean )
    df['pm10'] = df['pm10'].fillna( pm10_mean )
    df['o3'] = df['o3'].fillna( o3_mean )
    df['no2'] = df['no2'].fillna( no2_mean )
    df['so2'] = df['so2'].fillna( so2_mean )
    df['co'] = df['co'].fillna( co_mean )

    df['aqi'] = df[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']].max(axis=1)

    return df

def generate_insight_plots(df):
    plots = []
    columns = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    for col in columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='month', y=col, data=df, ax=ax)
        
        median_value = df[col].median()
        max_value = df[col].max()
        max_date = df.loc[df[col].idxmax(), 'date']
        
        ax.set_title(f'{col.upper()} over time (Median: {median_value}, Max: {max_value} on {max_date})')
        ax.set_xlabel('Date(Months)')
        ax.set_ylabel(col.upper())
        
        plots.append(fig)
        plt.close(fig)
        
    return plots

def normalize(df):

    x = np.array(range(len(df)))
    y = df['aqi'].values

    s1 = StandardScaler()
    s2 = StandardScaler()

    X_norm = s1.fit_transform(x.reshape(-1, 1))
    Y_norm = s2.fit_transform(y.reshape(-1, 1))

    return x,y,X_norm,Y_norm,s1,s2

def test_train_split(x,y,X_norm,Y_norm):
    
    np.random.seed(42)
    train_idx = np.random.choice(range(len(X_norm)), size=int(len(X_norm) * 0.7), replace=False)
    test_idx = list(set(range(len(X_norm))) - set(train_idx))

    X_train = x[train_idx]
    y_train = y[train_idx]

    X_test = x[test_idx]
    y_test = y[test_idx]

    X_norm_train = X_norm[train_idx]
    y_norm_train = Y_norm[train_idx]

    X_norm_test = X_norm[test_idx]
    y_norm_test = Y_norm[test_idx]

    return X_train,y_train,X_test,y_test,X_norm_train,y_norm_train,X_norm_test,y_norm_test


# def plot_fit_predict(model, X_norm_train, y_norm_train, X_norm_test, y_norm_test, X_lin, title, s1,s2 plot=False):

#     model.fit(X_norm_train, y_norm_train)

#     y_hat_train = model.predict(X_norm_train).reshape(-1, 1)
#     y_hat_test = model.predict(X_norm_test).reshape(-1, 1)

#     # Transform back to original scale
#     y_hat_train = s2.inverse_transform(y_hat_train)
#     y_hat_test = s2.inverse_transform(y_hat_test)

#     y_hat_lin = s2.inverse_transform(model.predict(X_lin).reshape(-1, 1))

#     errors[title] = {"train": mean_squared_error(y_train, y_hat_train),
#                      "test": mean_squared_error(y_test, y_hat_test)}

#     if plot:
#         plt.plot(X_train, y_train, 'o', label='train', ms=1, color='blue')
#         plt.plot(X_test, y_test, 'o', label='test', ms=2, color='lightgreen')
#         plt.plot(s1.inverse_transform(X_lin_1d.reshape(-1, 1)), y_hat_lin, label='model', ms=4,color='darkred')

#         plt.xlabel('Months since first measurement')
#         plt.ylabel('AQI Levels')
#         plt.legend()
#         plt.title('{}\n Train MSE: {:.2f} | Test MSE: {:.2f}'.format(title, errors[title]["train"], errors[title]["test"]))

#     return errors[title]

