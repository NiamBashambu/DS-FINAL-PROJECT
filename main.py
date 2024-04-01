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
FILENAME = "AAPL.csv"
YEAR_HEADER = "Date"

def main():
    
   
# Get a list of all CSV files in a directory
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
    # Print the pivoted dataframe
    print(pivoted_df)
    
# Print the combined dataframe
    

    #given the combined dataset we have here create model to predict stocks later down the line
    
    #take the data from a few stocks, UBER, AAPL, GOOG, NIKE


    uber_csv = pivoted_df["UBER (1)"]

    apple_csv = pivoted_df["AAPL"]

    google_csv = pivoted_df["GOOG"]

    nike_csv = pivoted_df["NKE"]
    
    '''
    #print(uber_csv)
    dataframes = []
    dataframes.append(uber_csv)
    dataframes.append(apple_csv)
    dataframes.append(google_csv)
    dataframes.append(nike_csv)
    for dataframe in dataframes:
        plt.figure(figsize = (18,9))
        plt.plot(dataframe)
        plt.title(dataframe)
    #plt.xticks(range(0,1827))
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('CLose Price',fontsize=18)
        plt.show()
    '''
    
    
    
    
    ### this is a logistic regression model that converst the values of hte price to 0 and 1 determine if the stock price will increase or decrease
    stocks = ['AAPL', 'GOOG', 'UBER (1)', 'NKE']
    for stock in stocks:
    
        pivoted_df[stock + '_Diff'] = pivoted_df[stock].diff()
    
        pivoted_df[stock + '_Target'] = (pivoted_df[stock + '_Diff'] > 0).astype(int)

#  AAPL
    X = pivoted_df[['AAPL']].iloc[1:]  
    y = pivoted_df['AAPL_Target'].iloc[1:] 

# split into train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create and train logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

# evaluate said model
    predictions = lr.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))

    predictions_proba = lr.predict_proba(X_test)[:, 1]  #get probability

# creating a scatter plot with the probabilities
    plt.figure(figsize=(10, 6))

# ploting actuall outcomes
    plt.scatter(X_test, y_test, color='black', label='Actual Outcomes (0: Decrease, 1: Increase)')

# plot the predicted outcomes
    plt.scatter(X_test, predictions_proba, color='blue', alpha=0.5, label='Predicted Probabilities of Increase')

    plt.title('Actual Outcomes vs. Predicted Probabilities for AAPL Stock')
    plt.xlabel('Previous Day Closing Price')
    plt.ylabel('Outcome / Predicted Probability')
    plt.legend()
    plt.show()
    #trying to do pairplot(don't do it it lags out your computer cuz the data is too big)
    #sns.pairplot(pivoted_df)
    #plt.show()

    #making the linear regression model to show the predicted prices
    
    pivoted_df['AAPL_lagged'] = pivoted_df['AAPL'].shift(1)
    pivoted_df.dropna(inplace=True)  # Drop the first row which now contains NaN

# Features and target variable
    X = pivoted_df[['AAPL_lagged']]  # Features: Previous day's closing price
    y = pivoted_df['AAPL']
    

# split into train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    actual = y_test
    #print(predictions)

    

    #print(mean_squared_error(predictions,actual))

    #need to make the graphs


#Comparison of both linear regression in one graph
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', label='Actual Prices')
    plt.plot(X_test, predictions, color='blue', linewidth=2, label='Predicted Prices')
    plt.title('Actual vs Predicted AAPL Stock Prices')
    plt.xlabel('Previous Day Closing Price')
    plt.ylabel('Next Day Closing Price')
    plt.legend()
    plt.show()  

'''
#Bar Chart of predictions and acutual bar graph rate of stock
    plt.bar(X_test, y_test,color = 'blue')
    plt.bar(X_test,predictions,color = "pink")
    plt.title('Predicted Outcomes and Probabilites for AAPL Stock')
    plt.xlabel('Previous Day Closing Price')
    plt.ylabel('Actual Outcome')
    plt.legend()
    plt.show()
'''

if __name__ == "__main__":
    main()


