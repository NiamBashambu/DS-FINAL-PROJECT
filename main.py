import csv
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sys
import os

from utils import get_filename, read_csv, lst_to_dct, normalize, moving_avg

DIR = "/Users/niambashambu/Desktop/DS FINAL PROJECT/data"
FILENAME = "AAPL.csv"



def get_filename(dirname, ext = ".csv"):
    '''give a directory name (string), 
    return the full path and name 
    for every non directory file in the directory (list of strings)
    '''
    filenames = []
    files = os.listdir(dirname)
    for file in files:
        full_path = os.path.join(dirname, file)  # Corrected to use the full path
        if not os.path.isdir(full_path) and file.endswith(ext):  # Corrected check
            filenames.append(full_path) 
    return filenames
def main():
    data = read_csv(DIR + '/'+FILENAME)
    dct = lst_to_dct(data)

    filenames = get_filename(DIR)
    print(filenames)

    names = []
    for filename in filenames:
        lst = read_csv(filename)
        dct = lst_to_dct(lst)




if __name__ == "__main__":
    main()
