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

#LOAD THE DATASET

#load the data 
df = pd.read_csv("evaluation_data.csv")
df.info()

#Plotting countplot of clusters
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x=df["Clusters"], palette= pal)
pl.set_title("Distribution Of The Clusters")
plt.show()

pl = sns.scatterplot(data = df,x=df["total_spend"], y=df["Income"],hue=df["Clusters"], palette= pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

plt.figure()
pl=sns.swarmplot(x=df['Clusters'], y=df['total_spend'],color=  "#CBEDDD", alpha=0.5)
pl=sns.boxenplot(x=df["Clusters"], y=df["total_spend"], palette=pal)
plt.show()

#creating a feature to get sum of accepted promotions 
df['Total_promo'] = df['AcceptedCmp1']  + df['AcceptedCmp2'] + df['AcceptedCmp3']+ df['AcceptedCmp4'] + df["AcceptedCmp5"]
#plot of total campaign accepted
plt.figure()
pl=sns.countplot(x=df['Total_promo'] , hue=df['Clusters'],palette= pal)
pl.set_title("Count of Promo accepted")
pl.set_xlabel("Number of total accepted promotions")
plt.show()

#plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y= df['NumDealsPurchases'], x=df['Clusters'],palette=pal)
pl.set_title("Number of deals purhase")
plt.show()


#for more details in purhasing style 
places = ['NumWebVisitsMonth',  'NumWebPurchases','NumCatalogPurchases','NumStorePurchases']

for i in places:
   plt.figure()
   sns.jointplot(x= df[i],y=df['total_spend'],hue= df["Clusters"],palette= pal)
   plt.show()

   Personal =  [ "Kidhome","Teenhome","Costumer_for", "age ", "Children", "Family_Size", "is_parent", "Education","living_with"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=df[i], y=df["total_spend"],hue=df["Clusters"],kind="kde", palette=pal)
    plt.show()