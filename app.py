import main as mn
from model1 import MODEL1
import pandas as pd 
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#title and intro of project

st.title("Location specific Air Quality Analysis and Regressor (Mumbai)")
st.header("Introduction and Motivation")

txt = '''
Welcome to my Data Science and Machine Learning project!

My name is Aryan Solanki, and I am an AI major at IIT Gandhinagar. You can find my work on my [GitHub](https://google.com).

To gain practical experience, I chose to work with the [World Air Quality Dataset](https://aqicn.org/historical/#city:india/mumbai) for Mumbai. This real-life dataset, with its anomalies and outliers, provides a challenging and insightful experience beyond the conventional Kaggle datasets.

In this project, I aim to:
- Analyze the air quality of 8 regions in Mumbai.
- Create basic plots of normalized hazardous gas concentration levels.
- Perform RBF Kernel-based (Fourier) interpolation.
- Develop an extrapolation model for the next two years.

Additionally, this project aims to raise awareness about the rising air pollution and its threat to air quality.

Thank you for exploring this project. I hope it offers valuable insights into air quality issues and the power of data science.
'''

st.write(txt)


st.image("image.png",caption="Description of Air Quality Levels")

#Region selection 

st.header("Region Selection and preliminary analysis")
st.write("Note: after selection of a location, if you intend to view for another location, preferbly do a hard refresh and then select.")

labels = ["Worli, Mumbai", "Mazgaon, Mumbai", "Colaba, Mumbai", "Sion, Mumbai", "Powai, Mumbai", "Nerul, Navi Mumbai"]

file_names = ["worli.csv", "mazgaon.csv", "colaba.csv", "sion.csv", "powai.csv", "nerul.csv"]

# Folder path
folder_path = "Datasets(Mumbai)/"

for label, file_name in zip(labels, file_names):
    
    if st.checkbox(label):
        full_path = folder_path + file_name
        df = mn.load_pre_process_calc_aqi(full_path)
        st.dataframe(df)

        def preliminary(df, label):
            # List of pollutants
            pollutants = ['so2', 'co', 'no2', 'pm25', 'pm10', 'o3']
            
            # Create a list to hold the summary data
            summary_data = {
                "Pollutant": [],
                "Highest Value": [],
                "Date of Highest Value": [],
                "Median Value": [],
            }
            
            for pollutant in pollutants:
                max_value = df[pollutant].max()
                max_date = df[df[pollutant] == max_value]['date'].values[0]
                median_value = df[pollutant].median()
                
                # Append data to summary_data dictionary
                summary_data["Pollutant"].append(pollutant)
                summary_data["Highest Value"].append(max_value)
                summary_data["Date of Highest Value"].append(max_date)
                summary_data["Median Value"].append(median_value)
            
            # Create a new DataFrame from the summary data
            summary_df = pd.DataFrame(summary_data)
            return summary_df


        summary_df = preliminary(df,label)

        st.subheader("\n\n Stats summary Dataframe")
        st.dataframe(summary_df, use_container_width=True)

        #preliminary plots/exploration, region specific insights 

        st.header("Trends on the AQI base gases")

        plot_list = mn.generate_insight_plots(df)


        def display_plots(plot_list):
            fig, axs = plt.subplots(2, 3, figsize=(20, 12))
            
            for i, plot in enumerate(plot_list):
                row = i // 3
                col = i % 3
                fig_temp = plot
                ax_temp = fig_temp.axes[0]
                
                # Render the individual plot into the subplot
                for line in ax_temp.get_lines():
                    axs[row, col].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color='red',ms=0.5)
                
                axs[row, col].set_title(ax_temp.get_title())
                axs[row, col].set_xlabel(ax_temp.get_xlabel())
                axs[row, col].set_ylabel(ax_temp.get_ylabel())
                
                # Hide the axis if desired
                # axs[row, col].axis('off')
                
            plt.tight_layout()
            
            # Display the combined figure using Streamlit
            st.pyplot(fig) 

        display_plots(plot_list)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        #normalization and aqi plot of normalized features. 

        st.header("Scatter Plot of AQI(Air Quality Index) ")
        import streamlit as st

        st.write("""
        ### How AQI is Calculated from the Six Base Pollutants

        The Air Quality Index (AQI) is a measure used to communicate how polluted the air currently is or how polluted it is forecast to become. The AQI is calculated for six major air pollutants regulated by the Clean Air Act:

        1. **Ground-level ozone (O3)**
        2. **Particulate matter (PM2.5 and PM10)**
        3. **Carbon monoxide (CO)**
        4. **Sulfur dioxide (SO2)**
        5. **Nitrogen dioxide (NO2)**

        The AQI is calculated as follows:

        1. **Data Collection**: Concentrations of the six pollutants are measured and reported in micrograms per cubic meter (µg/m³) for particulate matter and parts per million (ppm) or parts per billion (ppb) for gases.

        2. **Concentration to AQI Conversion**: Each pollutant's concentration is converted to a corresponding AQI value using standardized equations provided by environmental agencies like the EPA. These equations take into account health-based thresholds.

        3. **Final AQI Determination**: The AQI for the day is determined by the highest AQI value among the six pollutants. This ensures that the reported AQI reflects the pollutant with the greatest impact on health at that time.

        **Note**: The data source we are using has pre-processed the pollutant concentrations, meaning the concentrations are ready for direct use in AQI calculations. This simplifies our task to just applying the conversion equations and determining the maximum AQI value.

        For example, if the concentrations of pollutants are:
        - O3: 0.070 ppm
        - PM2.5: 55 µg/m³
        - PM10: 154 µg/m³
        - CO: 9 ppm
        - SO2: 75 ppb
        - NO2: 100 ppb

        We convert each to their respective AQI values, compare them, and the highest value is the AQI for that time period.
        """)


        x,y,X_norm,Y_norm,s1,s2 = mn.normalize(df)

        X_train,y_train,X_test,y_test,X_norm_train,y_norm_train,X_norm_test,y_norm_test = mn.test_train_split(x,y,X_norm,Y_norm)

        def plot_only (X_norm_train,y_norm_train,X_norm_test,y_norm_test):

            plt.figure(figsize=(10, 6))
            plt.plot(X_norm_train, y_norm_train, 'o', label='train',markersize=3)
            plt.plot(X_norm_test, y_norm_test, 'o', label='test', ms=3)
            plt.xlabel('(Normalized) Months since first measurement')
            plt.ylabel('(Normalized) AQI Level')
            plt.legend()
            plt.tight_layout()

            plt.show()
            st.pyplot() 

        plot_only(X_norm_train,y_norm_train,X_norm_test,y_norm_test) 

        #Interpolation RBF Fourier Series 
        st.header("Interpolation")

        obj = MODEL1([x, y, X_norm, Y_norm, s1, s2], [X_train, y_train, X_test, y_test, X_norm_train, y_norm_train, X_norm_test, y_norm_test])

        model_lr = LinearRegression()

        X_lin_1d = np.linspace(X_norm.min(), X_norm.max(), 100).reshape(-1, 1)

        best_params, lowest_mse = obj.hyperparam_selection_rbf(X_norm_train, y_norm_train, X_norm_test, y_norm_test, X_lin_1d)
        gamma = best_params['gamma']
        num_features = best_params['num_features']

        Xf_norm_train = obj.create_rff(X_norm_train.reshape(-1, 1), gamma, num_features)
        Xf_norm_test = obj.create_rff(X_norm_test.reshape(-1, 1), gamma, num_features)
        X_lin_rff = obj.create_rff(X_lin_1d, gamma, num_features)

        obj.plot_fit_predict(model_lr, Xf_norm_train, y_norm_train, Xf_norm_test, y_norm_test, 
                            X_lin_rff, f"Random Fourier Features (gamma={gamma}, NUM_features={num_features})", plot=True)
