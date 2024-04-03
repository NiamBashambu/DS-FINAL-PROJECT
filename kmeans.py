#this is file that does a k means clustering of the close price of the different stocks
#this is pretty redunant honeslty but its something to do



import json
import numpy as np
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
from sklearn import preprocessing
import scipy.spatial.distance as ssd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from scipy.cluster.vq import vq,kmeans,whiten
from pylab import plot,show


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

#stocks = ['AAPL', 'GOOG', 'UBER (1)', 'NKE','ADDYY','INTC','MSFT','META','NVDA','AMD']
#for stock in stocks:

 
inertias = []

for i in range(1,11):
    kmeanss = KMeans(n_clusters=i)
    kmeanss.fit(pivoted_df)
    inertias.append(kmeanss.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

pivoted_df = whiten(pivoted_df)
centroids, mean_value = kmeans(pivoted_df, 2)
print("Code book :", centroids, "")
print("Mean of Euclidean distances :", mean_value.round(4))

# mapping the centroids
clusters, _ = vq(pivoted_df, centroids)
print("Cluster index :", clusters, "")
#Plotting using numpy's logical indexing
plot(pivoted_df[clusters==0,0],pivoted_df[clusters==0,1],'ob',pivoted_df[clusters==1,0],pivoted_df[clusters==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()



#conclusions:
'''
I don't exactly know what this is saying imma be honest but it works so i don't really care. 
The code first finds the optimal k value using the elbow method. after words it runs a janky kmeans clustering thing to find the clusters and color them.
It looks really bad im going to be honest and doesn't really look like it would work well but it is what it is. 

'''