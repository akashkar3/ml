#1. FILE HANDLING (WRITE + READ)
# Exploring file properties
f = open("foo.txt", "w")
print(f.name)
print(f.closed)
print(f.mode)
f.close()

# Writing to file
file = open("sample.txt", "w")
file.write("EDA is good\n")
file.write("All are good")
file.close()

# Reading file
file = open("sample.txt", "r")
print(file.read())
file.close()

#2. PANDAS (CREATE + SAVE + READ)
import pandas as pd

data = {
    "Name": ["A", "B", "C"],
    "Age": [10, 20, 30],
    "Job": ["Accountant", "CEO", "Marketing"],
    "Salary": [50000, 100000, 60000]
}

df = pd.DataFrame(data)

# Save CSV
df.to_csv("sample.csv", index=False)

# Read CSV
df_read = pd.read_csv("sample.csv")
print(df_read)

# Save Excel
df.to_excel("sample.xlsx", index=False)

# Read Excel
df_excel = pd.read_excel("sample.xlsx")
print(df_excel)

#3. JSON FILE
import json

data = {
    "Name": "Akash",
    "Age": 21,
    "City": "Mumbai"
}

# Write JSON
with open("data.json", "w") as file:
    json.dump(data, file)

# Read JSON
with open("data.json", "r") as file:
    print(json.load(file))

#4. PICKLE FILE
import pickle

data = [1, 2, 3, 4]

# Write pickle
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)

# Read pickle
with open("data.pkl", "rb") as file:
    print(pickle.load(file))

#5. API USING REQUESTS
import requests

# Wrong API (example)
response = requests.get("http://api.open-notify.org/this-api")
print(response.status_code)

# Correct API
resp = requests.get("http://api.open-notify.org/astros")
print(resp.status_code)
print(resp.json())

#6. API REQUESTS (GET, POST, DELETE, HEAD)
import requests

# GET request
get_resp = requests.get("https://jsonplaceholder.typicode.com/posts")
print("GET Status:", get_resp.status_code)

# POST request
post_resp = requests.post(
    "https://jsonplaceholder.typicode.com/posts",
    json={
        "id": 1,
        "title": "updated",
        "body": "hello",
        "userId": 1
    }
)
print("POST Status:", post_resp.status_code)

# DELETE request
delete_resp = requests.delete("https://jsonplaceholder.typicode.com/posts/1")
print("DELETE Status:", delete_resp.status_code)

# HEAD request
head_resp = requests.head("https://jsonplaceholder.typicode.com/posts/1")
print("HEAD Status:", head_resp.status_code)

#7.WEB SCRAPING (BeautifulSoup)
import requests
from bs4 import BeautifulSoup

# Fetch webpage
res = requests.get("https://www.geeksforgeeks.org/java/java/")
soup = BeautifulSoup(res.content, "html.parser")

# Pretty print (optional)
#print(soup.prettify())

# Find article content
content = soup.find("div", class_="article--viewer-content")

if content:
    for para in content.find_all("p"):
        print(para.text.strip())
else:
    print("No article content found")

#8. detecting and removing outliers(Random) 
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest

# ---------------- DATA ----------------
data = pd.DataFrame({
    'X': [10, 12, 11, 13, 100, 14, 12],
    'Y': [20, 22, 21, 23, 200, 24, 22]
})

print("Original Data:\n", data)

# ---------------- 1. IQR METHOD ----------------
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

iqr_outliers = data[((data < lower) | (data > upper)).any(axis=1)]
print("\nIQR Outliers:\n", iqr_outliers)

# ---------------- 2. Z-SCORE METHOD ----------------
z_scores = np.abs(stats.zscore(data))
z_outliers = data[(z_scores > 3).any(axis=1)]
print("\nZ-score Outliers:\n", z_outliers)

# ---------------- 3. MAHALANOBIS DISTANCE ----------------
mean = data.mean()
cov = np.cov(data.values.T)
inv_cov = np.linalg.inv(cov)

data['Mahalanobis'] = data.apply(lambda row: mahalanobis(row[['X','Y']], mean, inv_cov), axis=1)

threshold = np.mean(data['Mahalanobis']) + 2*np.std(data['Mahalanobis'])
mahal_outliers = data[data['Mahalanobis'] > threshold]

print("\nMahalanobis Distances:\n", data['Mahalanobis'])
print("\nMahalanobis Outliers:\n", mahal_outliers)

# ---------------- 4. ISOLATION FOREST ----------------
model = IsolationForest(contamination=0.2, random_state=42)
data['Isolation'] = model.fit_predict(data[['X','Y']])

iso_outliers = data[data['Isolation'] == -1]
print("\nIsolation Forest Outliers:\n", iso_outliers)

#9.IQR METHOD + BOXPLOT
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Marks': [45, 50, 52, 48, 47, 90, 49, 46, 51, 44]}
df = pd.DataFrame(data)

# Boxplot
sns.boxplot(y=df['Marks'])
plt.show()

# IQR
Q1 = df['Marks'].quantile(0.25)
Q3 = df['Marks'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[(df['Marks'] >= lower) & (df['Marks'] <= upper)]
print("IQR Clean Data:\n", df_clean)

#10. Z-score method
from scipy import stats

data = {'values': [12, 15, 14, 10, 9, 100, 8, 13, 10, 14, 11, 10]}
df = pd.DataFrame(data)

z = abs(stats.zscore(df['values']))

df_clean = df[z <= 2]
print("Z-score Clean Data:\n", df_clean)

#11.MULTIVARIATE Z-SCORE (X, Y)
from scipy.stats import zscore

data = {
    'x': [10, 12, 11, 13, 12, 50],
    'y': [20, 22, 21, 23, 22, 100]
}

df = pd.DataFrame(data)

df['z_x'] = zscore(df['x'])
df['z_y'] = zscore(df['y'])

df_clean = df[(abs(df['z_x']) <= 2) & (abs(df['z_y']) <= 2)]

plt.scatter(df_clean['x'], df_clean['y'])
plt.title("Cleaned Scatter Plot")
plt.show()

#12.GRUBBS TEST
import numpy as np
from outliers import smirnov_grubbs as grubbs

data = np.array([5, 14, 15, 15, 14, 19, 17, 16, 20, 22, 84, 28, 11, 9, 29, 14])

max_outlier = grubbs.max_test(data, alpha=0.05)
print("Grubbs Outlier:", max_outlier)

#13.MAHALANOBIS DISTANCE
from scipy.spatial.distance import mahalanobis
import numpy as np

data = {
    'feature1': [10, 12, 10, 14, 100, 20, 100],
    'feature2': [20, 24, 20, 28, 32, 30, 110]
}

df = pd.DataFrame(data)

mean = df.mean()
cov = np.cov(df.values.T)
inv_cov = np.linalg.inv(cov)

distances = []

for _, row in df.iterrows():
    dist = mahalanobis(row, mean, inv_cov)
    distances.append(dist)

df['Mahalanobis'] = distances

threshold = np.percentile(distances, 95)

df['Outlier'] = df['Mahalanobis'] > threshold
print(df)

#14. Isolation forest
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data[['sepal length (cm)', 'sepal width (cm)']]

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

pred = model.predict(X)

X['outlier'] = pred

print(X[X['outlier'] == -1])  # Outliers

#15.LOCAL OUTLIER FACTOR (LOF)
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

labels = lof.fit_predict(X)

X['LOF_outlier'] = labels

print(X[X['LOF_outlier'] == -1])

#16.ONE-CLASS SVM
from sklearn.svm import OneClassSVM

model = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')

model.fit(X[['sepal length (cm)', 'sepal width (cm)']])

labels = model.predict(X[['sepal length (cm)', 'sepal width (cm)']])

X['SVM_outlier'] = labels

print(X[X['SVM_outlier'] == -1])

#17.Data analysis
import pandas as pd

# Load dataset
df = pd.read_csv("salary.csv")

# View data
print(df.head())      # first 5 rows
print(df.tail())      # last 5 rows

# Basic info
print(df.info())
print(df.dtypes)

# Size & structure
print(df.shape)       # rows, columns
print(len(df))        # number of rows
print(df.columns)

# Statistics
print(df.describe())

# Column operations
print(df['salary'].mean())
print(df['salary'].std())
print(df['salary'].count())

#18.Data analysis(FILTERING + GROUPBY + INDEXING)
# Groupby
print(df.groupby('Age')['salary'].mean())

# Filtering
high_salary = df[df['salary'] > 20000]
print(high_salary)

dentists = df[df['Job'] == 'Dentist']
print(dentists)

# Slicing
print(df[10:20])

# iloc (position-based)
print(df.iloc[10:20, [1,3]])
print(df.iloc[-1])   # last row

# Sorting
print(df.sort_values(by='salary'))
print(df.sort_values(by=['salary','Age'], ascending=[True, False]))

#19. check missing values
import pandas as pd

df = pd.read_csv("salary.csv")

# Check null values (True/False)
print(df.isnull())

# Row-wise check (any null in row)
print(df.isnull().any(axis=1))

# Column-wise count of nulls
print(df.isnull().sum())

# Row-wise count of nulls
print(df.isnull().sum(axis=1))

# Total null values in dataset
print(df.isnull().sum().sum())

#20. Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.hist(df['salary'], bins=10)
plt.title("Histogram of Salary")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

#Boxplot
sns.boxplot(x=df['salary'])
plt.title("Boxplot of Salary")
plt.show()

#Violin Plot
sns.violinplot(x=df['Age'])
plt.title("Violin Plot of Age")
plt.show()

#Scatter Plot
plt.scatter(df['Age'], df['salary'])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()

#Bar Plot
sns.barplot(x='Job', y='salary', data=df)
plt.title("Job vs Salary")
plt.show()

#Violin Plot (Job vs Salary)
sns.violinplot(x='Job', y='salary', data=df)
plt.title("Job vs Salary")
plt.show()

#Pie Chart
df['Job'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Job Distribution")
plt.show()


