# d10
#scikit-learn

from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt

X,y = make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,
                          random_state=22,n_clusters_per_class=1,scale=100))
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

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
