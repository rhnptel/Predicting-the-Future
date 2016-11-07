import pandas as pd 
import matplotlib.pyplot as plt
from pyplot import plot, show
import numpy as np 
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.spatial.distance import cdist
from operator import itemgetter


iris = pd.read_csv('/Users/rohanpatel/Downloads/iris.data.csv', names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width' 'class'])

iris['class'] = pd.Categorical(iris['class']).labels


# Generate scatterplots to view clusters

plt.scatter(iris['sepal_length'], iris['sepal_width'], c=iris['class'])
plt.scatter(iris['petal_width'], iris['petal_length'], c=iris['class'])
plt.scatter(iris['petal_length'], iris['sepal_length'], c=iris['class'])
plt.scatter(iris['petal_width'], iris['sepal_length'], c=iris['class'])
plt.scatter(iris['petal_length'], iris['sepal_width'], c=iris['class'])

#petal width and sepal length seem to be the only one that doesn't cluster well




