#this file is for when a user takes an input of a stock out of our 10 stocks and it gives them whether or not they should
#invest based on how the stock has done in the past


import csv
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sys
import os
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


from utils import get_filename, read_csv, lst_to_dct, normalize, moving_avg,mse

DIR = "/Users/niambashambu/Desktop/DS FINAL PROJECT/data"


csv_files = glob.glob('data/*.csv')
    
    

# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['filename'] = os.path.basename(csv_file).replace('.csv', '')  # Add stock name as a column
        
    combined_df = pd.concat([combined_df, df[['Date', 'Close', 'filename']]])
    
    # Pivot the DataFrame to have dates as index and stocks as columns
pivoted_df = combined_df.pivot(index='Date', columns='filename', values='Close')
pivoted_df= pivoted_df.fillna(0)

print(pivoted_df)


#taking the linear regression model from before but changing it slightly to accomodate for all stocks


stocks = ['AAPL', 'GOOG', 'UBER (1)', 'NKE','ADDYY','INTC','MSFT','META','NVDA','AMD']
for stock in stocks:
    pivoted_df[stock + '_lagged'] = pivoted_df[stock].shift(1)

pivoted_df.dropna(inplace=True) 
print(pivoted_df) # Drop the first row which now contains NaN

stock_variable = input("Gimme a stock avlue that u want to do the model on ")

# Features and target variable
X = pivoted_df[[stock_variable+'_lagged']]  # Features: Previous day's closing price
y = pivoted_df[stock_variable]
    

# split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
actual = y_test
    #print(predictions)

    

print(mean_squared_error(predictions,actual))


plt.figure(figsize=(10, 6))
plt.scatter(X_test, actual, color='black', label='Actual Prices')
plt.plot(X_test, predictions, color='blue', linewidth=2, label='Predicted Prices')
plt.title('Actual vs Predicted  '+ stock_variable+' Stock Prices')
plt.xlabel('Previous Day Closing Price')
plt.ylabel('Next Day Closing Price')
plt.legend()
plt.show()  



