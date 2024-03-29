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
    

    print(uber_csv)


    plt.figure(figsize = (18,9))
    plt.plot(uber_csv)
    #plt.xticks(range(0,1827))
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('CLose Price',fontsize=18)
    plt.show()

    #X_train , X_test, y_train, y_test = train_test_split(,range(0,1260))
    

    lr = LogisticRegression(max_iter=1000,C=1)
    lr.fit(X_train,y_train)

    predictions = lr.predict(X_test)
    predictions=predictions.tolist()
    actual = y_test
    1-mse(predictions,actual)
    

if __name__ == "__main__":
    main()


