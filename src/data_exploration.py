#import  libraries

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


data  = pd.read_csv("marketing_campaign-1.xls", sep="\t")
print("Number of data points :", len(data) )
data.head()

#information features
data.info()

#lets remove the misiing values 
data = data.dropna()
print("total number if data points after removing the null are ",len(data))


#CONVERTING DATA TO DATIME FORMAT
#convert the Dt_Customer column to datetime format
#and extract the date from it
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors='coerce')
#check if the conversion was successful
print("The data type of Dt_Customer after conversion is:", data["Dt_Customer"].dtype)
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors='coerce')
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)  
#Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in therecords:",max(dates))
print("The oldest customer's enrolment date in the records:",min(dates))

days = [] #list to store the days between the first/last activity and the recent date
d1 = max(dates) #the recent date 
for i in dates : # loop through the dates 
  delta = d1 - i # calulate the days between the recentg days and the  first/last activity
  days.append(delta)#store it 

data["Costumer_for"] = days
#convert the Costumer_for column to numeric format
data['Costumer_for'] = pd.to_numeric(data['Costumer_for'],errors='coerce')

print('total categories of the features Martal_Status:/n', data['Marital_Status'].value_counts())
print("total categories of the features Education:/n",data['Education'].value_counts())

# Set style
sns.set(style="whitegrid")

# Marital Status
plt.figure(figsize=(6,4))
sns.countplot(x="Marital_Status", data=data, order=data['Marital_Status'].value_counts().index)
plt.title("Distribution of Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.show()

# Education
plt.figure(figsize=(6,4))
sns.countplot(x="Education", data=data, order=data['Education'].value_counts().index)
plt.title("Distribution of Education Levels")
plt.xlabel("Education")
plt.ylabel("Count")
plt.show()
