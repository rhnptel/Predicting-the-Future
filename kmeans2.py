import pandas as pd
import numpy as np
from scipy.cluster import vq
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/rohanpatel/Downloads/un countries data.csv')

len(df)
#207

df.count()
#columns that we need have enough values
#207 countries

df.dtypes

k_list = range(1, 11)
dist_list = []
sub_df = vq.whiten(df[['lifeMale', 'GDPperCapita']].dropna())
for k in k_list:
    centroids, dist = vq.kmeans(sub_df, k)
    dist_list.append(dist)

plt.figure(figsize=(10,5))
plt.plot(k_list, dist_list)
plt.show()


k_list = range(1, 11)
dist_list = []
sub_df = vq.whiten(df[['lifeFemale', 'GDPperCapita']].dropna())
for k in k_list:
    centroids, dist = vq.kmeans(sub_df, k)
    dist_list.append(dist)

plt.figure(figsize=(10,5))
plt.plot(k_list, dist_list)
plt.show()


k_list = range(1, 11)
dist_list = []
sub_df = vq.whiten(df[['infantMortality', 'GDPperCapita']].dropna())
for k in k_list:
    centroids, dist = vq.kmeans(sub_df, k)
    dist_list.append(dist)

plt.figure(figsize=(10,5))
plt.plot(k_list, dist_list)

#kmeans male life expectancy, cluster size = 3
model = KMeans(3)
sub_df = df[['lifeMale', 'GDPperCapita']].dropna()
model.fit(sub_df)
y_pred = model.predict(sub_df)
plt.figure(figsize=(10, 5))
plt.scatter(sub_df['GDPperCapita'], sub_df['lifeMale'], c=y_pred)
plt.ylabel('lifeMale')
plt.xlabel('GDP per Capita')
plt.xlim(0)
plt.show()

#kmeans female life expectancy, cluster size = 3
model = KMeans(3)
sub_df = df[['lifeFemale', 'GDPperCapita']].dropna()
model.fit(sub_df)
y_pred = model.predict(sub_df)
plt.figure(figsize=(10, 5))
plt.scatter(sub_df['GDPperCapita'], sub_df['lifeFemale'], c=y_pred)
plt.ylabel('lifeFemale')
plt.xlabel('GDP per Capita')
plt.xlim(0)
plt.show()

#kmeans infant mortality, cluster size = 3
model = KMeans(3)
sub_df = df[['infantMortality', 'GDPperCapita']].dropna()
model.fit(sub_df)
y_pred = model.predict(sub_df)
plt.figure(figsize=(10, 5))
plt.scatter(sub_df['GDPperCapita'], sub_df['infantMortality'], c=y_pred)
plt.ylabel('Infant Mortality')
plt.xlabel('GDP per Capita')
plt.xlim(0)
plt.show()



