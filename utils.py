
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt

def read_csv(filename):
    ''' given the name of a csv file, return its contents as a 2d list,
        including the header.'''
    data = []
    
    with open(filename, "r") as infile:
        csvfile = csv.reader(infile)
        for row in csvfile:
            data.append(row)
    return data


def lst_to_dct(lst):
    dct = {}
   
    
    for header in lst[0]:
        dct[header]= []
    for row in lst[1:]:
        for i in range(len(row)):
            dct[lst[0][i]].append(row[i])
    return dct

def median(orig_lst):
    ''' given a list of numbers, compute and return median'''
    lst = orig_lst.copy()
    lst.sort()
    mid = len(lst) // 2
    if len(lst) % 2 == 1:
        return lst[mid]
    else:
        avg = (lst[mid] + lst[mid - 1]) / 2
        return avg

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

def normalize(lst):
    '''
    given a list of numbers, return a list of min/max normalized values
    
    '''
    minn = min(lst)
    maxx = max(lst)

    normal = []

    for x in lst:
        normal_x = (x-minn)/((maxx-minn))
        normal.append(normal_x)
    return normal

#fucntion call with the two parameters, one default
def moving_avg(lst,num=2):
    #creating new list for which to append the new values onto
    newlst = []
    #range is the leng of list minus the value of which numbers to average by + 1
    for i in range(len(lst)- num+1):
        #finding the sum of the terms in the num value
        num_sum = sum(lst[i:i+num])
        #findung the average
        num_avg = num_sum/num
        #adding new value to the list
        newlst.append(num_avg)

    return newlst
        