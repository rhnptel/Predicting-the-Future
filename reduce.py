import matplotlib.pyplot as plt
from numpy.random import random
import pandas as pd
from collections import Counter
from random import choice
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


iris = datasets.load_iris()
X = iris.data
Y = iris.target

target_names = iris.target_names

#Perform PCA on Iris Data
pca = PCA(n_components=2)
x = pca.fit(X).transform(X)

X[:3]
x[:3]

#Plot PCA Sample
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(x[y == i, 0], x[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA of IRIS dataset')

plt.show()
#very similar to the original

#Retraining KNN lesson 5
x, y = df.ix[choice(range(len(df)))][:2]

df_decomposed = pd.DataFrame(x)
distances = [((pair[0] - y)**2 + (pair[1] - x)**2)**(-1/2) for pair in zip(df_decomposed[0], df_decomposed[1])]
df_decomposed["distances"] = distances
df_decomposed["Species"] = df["Species"]

majority = list(Counter([pair[1] for pair in sorted(zip(df_decomposed.distances, df_decomposed.Species)
                                                  
majority 
#setosa
#majority class was virginica before, now it is setosa

#LDA on Iris Data
lda = LinearDiscriminantAnalysis(n_components=2)
a = lda.fit(X, y).transform(X)

#Plot LDA Sample
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(a[y == i, 0], a[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.show()

#Retrain KNN lesson 4
x, y = df.ix[choice(range(len(df)))][:2]

df_decomposed = pd.DataFrame(a)
distances = [((pair[0] - y)**2 + (pair[1] - x)**2)**(-1/2) for pair in zip(df_decomposed[0], df_decomposed[1])]
df_decomposed["distances"] = distances
df_decomposed["Species"] = df["Species"]

majority = list(Counter([pair[1] for pair in sorted(zip(df_decomposed.distances, df_decomposed.Species)

majority 
#virginica
#same class prediction



