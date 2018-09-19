# d10
#scikit-learn

from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt

X,y = make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,
                          random_state=22,n_clusters_per_class=1,scale=100))
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

from sklearn import processing
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC

X,y = make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,
                          random_state=22,n_clusters_per_class=1,scale=100))
X = processing.scale(X)     #normalization 标准化
X_train,X_test,y_train,y_test = train_test_split(X,y,test.size=.3)
df = SVC()
df.fit(X_train,y_train)
print(df.score(X_test,y_test))

from sklearn.datasets import loaded_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = loaded_iris
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4)
knn = KNeighborClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

from sklearn.datasets import loaded_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = loaded_iris
X = iris.data
y = iris.target

knn = KNeighborClassifier(n_neighbors=5)
scores = cross_val_score(knn,X,y,cv=5,scoring='accuracy')
print(scores) => [0.96666667 1.         0.93333333 0.96666667 1.        ]
print(scores.mean()) => 0.9733333333333334

import matplotlib.pyplot as plt
k_range = range(1,31)
k_score = []
for k in k_range:
    knn = KNeighborClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    k_scores.append(scores)
plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

digits = load_digits()
X = digits.data
y = digits.target

train_size,train_loss,test_loss = learning_curve(SVC(gamma=0.1),X,y,cv=10,scoring='neg_mean_squared_error',
    train_sizes=[0.1,0.25,0.5,0.75,1])
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)
plt.plot(train_size,train_loss,'o-',color='r',label='Training')
plt.plot(train_size,test_loss,'o-',color='g',label='Cross-validation')
plt.xlabel('Training examples')
plt.ylabel('Loss')
plt.legend()
plt.show()

#overfitting
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve

digits = load_digits()
X = digits.data
y = digits.target

train_loss,test_loss = learning_curve(SVC(),X,y,param_name='gamma',param_range=param_range,cv=10,scoring='neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)
plt.plot(param_range,train_loss,'o-',color='r',label='Training')
plt.plot(param_range,test_loss,'o-',color='g',label='Cross-validation')
plt.xlabel('gamma')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X = iris.data
y = iris.target
clf.fit(X,y)

#method 1: pickle
import pickle
#save
with open('save/clf.pickle','wb') as f:
    pickle.dump(clf,f)
#restore
with open('save/clf.pickle','rb') as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1])) => [0]

#method 2: joblib
from sklearn.externals import joblib
#save
joblib.dump('save/clf.pkl')
#restore
clf3 = joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1])) => [0]
