## Introduction:
This notebook demonstrates time series forecasting using the FBProphet library in Python. The dataset used in this project is the daily number of passengers for an airline company, which spans from 1949 to 1960. The goal of the project is to forecast the number of passengers for the next few years based on the historical data.

## Data:
The dataset used in this project can be found in the "airline_passengers.csv" file. It contains two columns: "Month" and "Passengers". The "Month" column represents the time period and is in the format "yyyy-mm". The "Passengers" column represents the number of passengers for that particular month.

## Methods:
The project follows the following steps:

1. Importing the necessary libraries and loading the dataset
2. Data preprocessing: parsing the "Month" column, renaming the columns, and converting the "Passengers" column to a numeric type.
3. Data visualization: plotting the time series data to visualize the trend and seasonality.
4.Modeling: creating a Prophet model and fitting it to the data. The model is trained on the first 80% of the data and tested on the remaining 20%.
5. Forecasting: using the model to forecast the number of passengers for the next few years.
6. Evaluation: evaluating the performance of the model using various metrics such as mean absolute error, mean squared error, and root mean squared error.
7. Visualization: plotting the forecasted data along with the historical data to visualize the accuracy of the model.
Results:
The project demonstrates that FBProphet can be used to effectively forecast time series data. The model was able to capture the trend and seasonality of the data and accurately forecast the number of passengers for the next few years. The evaluation metrics showed that the model performed well on the test data.

Conclusion:
FBProphet is a powerful tool for time series forecasting and can be used to generate accurate forecasts for various applications. With the proper data preprocessing and model tuning, FBProphet can provide valuable insights for decision-making.
