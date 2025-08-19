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

# 🔹 Load dataset
data  = pd.read_csv("marketing_campaign-1.xls", sep="\t")
print("Number of data points :", len(data) )
print(data.head())

# 🔹 Dataset info
data.info()

# 🔹 Remove missing values
data = data.dropna()
print("Total number of data points after removing nulls:", len(data))

# 🔹 Convert date column
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors='coerce')

# Extract dates
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)  

print("The newest customer's enrolment date in the records:", max(dates))
print("The oldest customer's enrolment date in the records:", min(dates))

# 🔹 Create 'Customer_for' feature
days = []
d1 = max(dates)
for i in dates:
    delta = d1 - i
    days.append(delta)

data["Customer_for"] = days
data["Customer_for"] = pd.to_numeric(data["Customer_for"], errors='coerce')

# 🔹 Value counts for categorical features
print("Total categories of the feature Marital_Status:\n", data['Marital_Status'].value_counts())
print("Total categories of the feature Education:\n", data['Education'].value_counts())

# 🔹 Visualization
sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.countplot(x="Marital_Status", data=data, order=data['Marital_Status'].value_counts().index)
plt.title("Distribution of Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Education", data=data, order=data['Education'].value_counts().index)
plt.title("Distribution of Education Levels")
plt.xlabel("Education")
plt.ylabel("Count")
plt.show()

# 🔹 Save cleaned dataset for feature engineering
data.to_csv("explored_data.csv", index=False)
print("✅ Data exploration complete. Cleaned dataset saved as 'explored_data.csv'")
 