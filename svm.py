from sklearn import datasets, svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


iris = datasets.load_iris()
data = iris.data
target = iris.target

#petal length and sepal width of three flowers
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

#The first 100 observations correspond to setosa and versicolor
plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()


svc = svm.SVC(kernel='linear')
X = iris.data[0:100, 1:3]
y = iris.target[0:100]
svc.fit(X, y)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

plot_estimator(svc, X, y)
#boundary is clean


#sepal length and sepal width
svc2 = svm.SVC(kernel = 'linear')
x2 = data[:, 0:2]
y2 = target
svc2.fit(x2, y2)
plot_estimator(svc2, x2, y2)

#sepal length and petal length
svc3 = svm.SVC(kernel = 'linear')
x3 = data[:, 0:3:2]
y3 = target
svc3.fit(x3, y3)
plot_estimator(svc3, x3, y3)

#petal length and petal width
svc4 = svm.SVC(kernel = 'linear')
x4 = data[:, 2:]
y4 = target
svc4.fit(x4, y4)
plot_estimator(svc4, x4, y4)


