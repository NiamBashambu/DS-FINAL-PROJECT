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

def clean_numeric(s):
    ''' given a string with extra characters $ or , or %, remove them
        and return the value as a float
    '''
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.replace(",", "")
    s= s.replace(":", "")
    return float(s)
def clean_data(dct):
    ''' given a dictionary that includes currency and
        numbers in the form x,xxx, clean them up and convert
        to int/float
    '''
    for key, value in dct.items():
        for i in range(len(value)):
            if not value[i].replace(" ", "").isalpha():
                value[i] = clean_numeric(value[i])

def main():
    

# Get a list of all CSV files in a directory
    csv_files = glob.glob('data/*.csv')
    

# Create an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

# Loop through each CSV file and append its contents to the combined dataframe
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
       
        df['filename'] = os.path.basename(csv_file)
        combined_df = pd.concat([combined_df, df])

# Print the combined dataframe
    print(combined_df)
    



if __name__ == "__main__":
    main()




#need to figure out a way to add the close price to the files and sync the dates
    #start cleaning - tj if u wanna do that