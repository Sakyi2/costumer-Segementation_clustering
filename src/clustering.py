#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)




# Load PCA dataset
PCA_ds = pd.read_csv("clustering_data.csv")
print(PCA_ds.head())

#load the data 
df = pd.read_csv("feature_engineered_data.csv")


#elbow describe the number of clusters u want
# we using the yellow brick elbow visualizer
print("Elbow method to determine the number of clusters to formed ")
Elbow_M = KElbowVisualizer(KMeans(),k=(10)) #create a visualizer from yellow brick runs from k1 to k2
Elbow_M.fit(PCA_ds)#fit data into the visualizer 
Elbow_M.show()#show the plot 


#initiating the agglomerative clustering
AC = AgglomerativeClustering(n_clusters=4)# u are asking ur algorithm to divied into groups
Yhat_AC = AC.fit_predict(PCA_ds)#fit the data into the model
PCA_ds['Clusters'] = Yhat_AC#assign the new column to the data frame
#Adding the Clusters feature to the orignal dataframe.
df["Clusters"]= Yhat_AC
print(Yhat_AC)

#visualizing the clusters
fig = plt.figure(figsize=(10,8))
ax= fig.add_subplot(111,projection = '3d',label = 'la')
ax.scatter(PCA_ds["col1"],PCA_ds["col2"],PCA_ds["col3"], s=40, c=PCA_ds["Clusters"], marker='o')
ax.set_title('plot of cluster')
plt.show()