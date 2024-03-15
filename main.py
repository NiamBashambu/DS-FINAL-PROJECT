import csv
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sys
import os
import pandas as pd
import glob

from utils import get_filename, read_csv, lst_to_dct, normalize, moving_avg

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
    
    # Print the pivoted dataframe
    print(pivoted_df)
    
# Print the combined dataframe
    

    





if __name__ == "__main__":
    main()


