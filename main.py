import csv
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import sys

from utils import get_filename, read_csv, lst_to_dct, normalize, moving_avg

DIR = "/Users/niambashambu/Desktop/DS FINAL PROJECT/data"
FILENAME = "AAPL.csv"

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
