from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class MODEL1:
    def __init__(self, list1, list2):
        [self.x, self.y, self.X_norm, self.Y_norm, self.s1, self.s2] = list1
        [self.X_train, self.y_train, self.X_test, self.y_test, 
         self.X_norm_train, self.y_norm_train, self.X_norm_test, self.y_norm_test] = list2
        
    def create_rff(self, X, gamma, NUM_features):
        rbf_feature = RBFSampler(gamma=gamma, n_components=NUM_features, random_state=1)
        X_features = rbf_feature.fit_transform(X)
        return X_features

    def plot_fit_predict(self, model, X_norm_train, y_norm_train, X_norm_test, y_norm_test, X_lin, title, plot=True):
        errors = {}
        model.fit(X_norm_train, y_norm_train)

        y_hat_train = model.predict(X_norm_train).reshape(-1, 1)
        y_hat_test = model.predict(X_norm_test).reshape(-1, 1)

        y_hat_train = self.s2.inverse_transform(y_hat_train)
        y_hat_test = self.s2.inverse_transform(y_hat_test)
        y_hat_lin = self.s2.inverse_transform(model.predict(X_lin).reshape(-1, 1))

        errors[title] = {
            "train": mean_squared_error(self.y_train, y_hat_train),
            "test": mean_squared_error(self.y_test, y_hat_test)
        }

        if plot:
            
            X_lin_1d = np.linspace(self.X_norm.min(), self.X_norm.max(), 100).reshape(-1, 1)
            plt.plot(self.X_train, self.y_train, 'o', label='train', ms=1, color='blue')
            plt.plot(self.X_test, self.y_test, 'o', label='test', ms=2, color='lightgreen')
            plt.plot(self.s1.inverse_transform(X_lin_1d.reshape(-1, 1)), y_hat_lin, label='model', ms=4, color='darkred')

            plt.xlabel('Months since first measurement')
            plt.ylabel('AQI Levels')
            plt.legend()
            plt.title("Interpolation of AQI vs Months")

            plt.show()
            st.pyplot() 
        
        return errors[title]

    def hyperparam_selection_rbf(self, X_norm_train, y_norm_train, X_norm_test, y_norm_test, X_lin_1d):
        gamma_values = np.arange(0, 101, 5)
        num_features_values = range(1, 100)

        best_params = None
        lowest_mse = float('inf')

        for gamma in gamma_values:
            for num_features in num_features_values:
                Xf_norm_train = self.create_rff(X_norm_train.reshape(-1, 1), gamma, num_features)
                Xf_norm_test = self.create_rff(X_norm_test.reshape(-1, 1), gamma, num_features)
                X_lin_rff = self.create_rff(X_lin_1d, gamma, num_features)
                
                model = LinearRegression()
                mse_dict = self.plot_fit_predict(
                    model, Xf_norm_train, y_norm_train, 
                    Xf_norm_test, y_norm_test, 
                    X_lin_rff, 
                    f"Random Fourier Features (gamma={gamma}, NUM_features={num_features})",
                    plot=False
                )
                
                train_mse = mse_dict['train']
                test_mse = mse_dict['test']
                
                if test_mse < lowest_mse:
                    lowest_mse = test_mse
                    best_params = {'gamma': gamma, 'num_features': num_features}

        print(f"Best parameters: gamma={best_params['gamma']}, num_features={best_params['num_features']}")
        print(f"Lowest MSE: {lowest_mse}")

        return best_params, lowest_mse
