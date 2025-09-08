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


# Load dataset saved from data evaluation
df = pd.read_csv("explored_data.csv")

print(f"✅ Loaded dataset with shape: {df.shape}")
df.head()

#create features
#1) age from birth

df['age '] = 2025 - df['Year_Birth']

#total spend on various items
spending_cols =['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts',"MntGoldProds"]
df['total_spend'] = df[spending_cols].sum(axis=1)

#living with feature from marital status 
df['living_with'] = df["Marital_Status"].replace({
    'Married': "Partner",
    "Single": "Alone",
    "Divorced":"Alone",
    "YOLO": "Alone",
    "Widow":"Alone",
    "Absurd":"Alone",
    'Together':'Partner'

})

#children (kids + teengager)
df['Children'] = df["Kidhome"] + df['Teenhome']

#total family size
df['Family_Size'] = df['Children'] + df['living_with'].replace({"Alone": 1, "Partner": 2})

#is_parent(binary_feature)
#👉 So, Is_Parent is just a binary feature (flag) that tells you if the customer is a parent or not, based on the Children column.
df['is_parent'] = np.where(df.Children >0,1,0)

#simpify eduactions
df["Education"] = df['Education'].replace({
    'Basic' : 'Undergraduate',
    "2n cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master":"Postgraduate",
    "PhD": "Postgraduate"

})


#Simply rename the long column names from the dataset into shorter, easier-to-read ones
df = df.rename(columns={
    "MntWines": "Wines",
    "MntFruits": "Fruits",
    "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish",
    "MntSweetProducts": "Sweets",
    "MntGoldProds": "Gold"
})

to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
df = df.drop(to_drop, axis=1)


df.describe()


#setting up color preference
sns.set(rc={'axes.facecolor':"#F5F5F5", 'figure.facecolor': "#DAD9D6CA"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(pallet)

# Features to plot
To_Plot = ["Income", "Recency", "Costumer_for", "age ", "total_spend", "is_parent"]

print("Relative Plot Of Some Selected Features: A Data Subset")

plt.figure()
sns.pairplot(df[To_Plot], hue= "is_parent",palette= (["#682F2F","#F3AB60"]))
#Taking hue 
plt.show()

#removing outliers
df = df[(df["age "] < 90)]
df = df[(df["Income"] < 600000)]

print("The total number of data-points after removing the outliers are:", len(df))


# Select only numeric columns
numeric_data = df.select_dtypes(include=[np.number])

# Correlation matrix
corrmat = numeric_data.corr()

plt.figure(figsize=(20,20))  
sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)
plt.show()
