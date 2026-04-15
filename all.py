#PCA(Principal Component Analysis)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

x=np.random.rand(100,3)

x_std=StandardScaler().fit_transform(x)

pca=PCA(n_components=2)

x_pca=pca.fit_transform(x_std)

print("Shape:",x_pca.shape)
print("Explained Variance:",pca.explained_variance_ratio_)

plt.plot(range(1,3),np.cumsum(pca.explained_variance_ratio_),marker="o")
plt.title("PCA")
plt.xlabel("Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.show()

#Randomized Search CV
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=SVC()

hyperparam_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf','poly'],
    'gamma':['scale','auto']
}

random_search=RandomizedSearchCV(estimator=model,param_distributions=hyperparam_grid,scoring='accuracy',cv=5,verbose=2,n_jobs=-1)
random_search.fit(x_train,y_train)

best_hyperparameters=random_search.best_estimator_
best_score=random_search.best_score_

print("Best model:",best_hyperparameters)
print("Best Score:",best_score)

best_model=random_search.best_estimator_
y_pred=best_model.predict(x_test)

print("Predictions:",y_pred)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

#Grid Search CV
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris= datasets.load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=SVC()
param_grid={
    'C':[1,10],
    'kernel':['linear','rbf','poly'],
    'gamma':['scale','auto']
}

grid=GridSearchCV(estimator=model,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,verbose=2,
                  n_jobs=-1)
grid.fit(x_train,y_train)

best_parameter=grid.best_estimator_
best_score=grid.best_score_

print(best_parameter)
print(best_score)

best_model=grid.best_estimator_
y_pred=best_model.predict(x_test)
print("Predictions",y_pred)

accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

#Underfitting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
x=np.linspace(0,10,100)
y=np.sin(x)+np.random.normal(0,0.3,size=x.shape)

x_reshaped=x.reshape(-1,1)

model=LinearRegression()
model.fit(x_reshaped,y)

y_pred=model.predict(x_reshaped)

plt.figure(figsize=(8,6))
plt.scatter(x,y,color='blue',label='data')
plt.plot(x,y_pred,color='red',label='Underfitting')

plt.title("Underfitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#Overfitting
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)

X=np.linspace(0,10,100)
y=np.sin(X)+np.random.normal(0,0.3,size=X.shape) #adds noise

X_reshape= X.reshape(-1,1)

model = make_pipeline(PolynomialFeatures(degree=10),LinearRegression())
model.fit(X_reshape,y)

y_pred= model.predict(X_reshape)

plt.figure(figsize=(8,6))
plt.scatter(X,y,color="blue",label="Data")
plt.plot(X,y_pred,color="red",label="overfit")
plt.title("Overfitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#Confusion Matrix (Binary)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

actual=np.array(['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
predicted=np.array(['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])

cm=confusion_matrix(actual,predicted)
print("Confusion matrix:\n",cm)

sns.heatmap(cm,annot=True,xticklabels=['Dog','Not Dog'],yticklabels=['Dog','Not Dog'])
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("Binary Confusion matrix")
plt.show()

print("classification_report:\n")
print(classification_report(actual,predicted))

#Confusion Matrix (Multiclass)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

actual=['cat']*10+['dog']*12+['horse']*10
predicted=['cat']*8+['dog']*2+['dog']*10+['horse']*2+['horse']*8+['dog']*2

classes=['cat','dog','horse']
cm=confusion_matrix(actual,predicted)
print("Confusion Matrix:\n",cm)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
disp.plot()
plt.title("Multiclass Confusion Matrix")
plt.show()

sns.heatmap(cm,annot=True,xticklabels=classes,yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multiclass Confusion Matrix")
plt.show()

print("Classification Report:\n")
print(classification_report(actual,predicted))

#Linear Regression
import matplotlib.pyplot as plt
from scipy import stats

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

slope,intercept,r,p,std_error=stats.linregress(x,y)

def myfunc(x):
    return slope*x + intercept

mymodel=list(map(myfunc,x))
yhat=myfunc(10)
print(mymodel)
print("Correlation Coefficient:",r)
print("Slope:",slope)
print("intercept",intercept)
print("predicted value at x=10 is",yhat)

plt.scatter(x,y)
plt.plot(x,mymodel)
plt.scatter(10,yhat)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()

#Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21]
y = [100,90,80,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

model=np.poly1d(np.polyfit(x,y,3))

yhat=model(10.5)

print("predicted value at x=10.5",yhat)

plt.scatter(x,y)
plt.plot(x,model(x))
plt.scatter(10.5,yhat)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polymonial Regression")
plt.show()

#K means clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [5, 6, 12, 5, 4, 13, 15, 7, 10, 14]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data=list(zip(x,y))

kmeans=KMeans(n_clusters=2,random_state=42)
kmeans.fit(data)

centroids=kmeans.cluster_centers_
labels=kmeans.labels_

print("centroids\n",centroids)
print("labels\n",labels)

plt.scatter(x,y,c=labels)
plt.scatter(centroids[:,0],centroids[:,1],c='red',s=50,alpha=0.75)
 
plt.title("K means clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Agglomerative Clustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering

x = [4, 6, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 21, 17, 16, 25, 24, 22, 21, 21]

data=list(zip(x,y))

linkage_data=linkage(data,method='complete',metric='euclidean')

dendrogram(linkage_data)
plt.title("dendogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

model=AgglomerativeClustering(n_clusters=2,linkage='complete')
labels=model.fit_predict(data)

print("cluster labels\n",labels)

plt.scatter(x,y,c=labels)
plt.title("Agglomerative Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

linkage_data = linkage(data, method='single', metric='euclidean')

dendrogram(linkage_data)
plt.title("Dendrogram (Single Linkage)")
plt.show()