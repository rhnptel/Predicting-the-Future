import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import DistanceMetric as dm
import collections


df = pd.read_csv('/Users/rohanpatel/downloads/iris.data.csv', names = ['sepal_length', 'sepal_width', 'petal_length', 'class'])

df['class'] = pd.Categorical(df['class']).codes

#scatterplots
plt.scatter(df['sepal_length'], df['sepal_width'], c=df['class'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

plt.scatter(df['sepal_width'], df['petal_length'], c=df['class'])
plt.show()

plt.scatter(df['petal_width'], df['sepal_length'], c=df['class'])
plt.show()

plt.scatter(df['petal_width'], df['sepal_length'], c=df['class'])
plt.show()

plt.scatter(df['petal_length'], df['sepal_width'], c=df['class'])
plt.show()

random_slength = random.uniform(4.0, 8.5)

random_swidth = random.uniform(1.5, 5.0)

print random_slength, random_swidth
#6.83085774014 3.35853306541

d_list = []
iris['dist'] = d_list
dist = dm.get_metric('euclidean')

for i,row in enumerate(range(len(iris))):
    d_list.append(dist.pairwise([[random_slength, random_swidth], [df['sepal_length'][i], df['sepal_width'][i]]])[0][1]



def knn(k):
    # Isolate nearest neighbors
    sub_iris = iris.sort(['dist'])[:k] 
    pass

    # Create counter
    class_count = collections.Counter()
    for n in list(sub_iris['class']):
        class_count[n] += 1
        
    # Find key with max val
    for key in class_count.keys():
        if class_count[key] == max(class_count.values()):
            print key, "is the majority fo the class subset."
            print "%s has %d of the %d points in the subset." % (key, max(class_count.values()), k)


knn(8)

#Iris-setosa is the majority fo the class subset.
#Iris-setosa has 5 of the 8 points in the subset.


