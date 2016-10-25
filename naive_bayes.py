import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB
import numpy as np

df = pd.read_csv('/Users/rohanpatel/Downloads/ideal_weight.csv')
df = df.rename(columns={"'actual'": "actual", "'ideal'": "ideal", "'diff'": "diff", "'id'": "id", "'sex'": "sex"})


df['sex'] = df['sex'].replace("'Male'", "Male")
df['sex'] = df['sex'].replace("'Female'", "Female")

df['sex'] = pd.Categorical(df['sex']).labels

df.groupby(['sex']).size()

df['actual'].hist()
df['ideal'].hist()
df['diff'].hist()
plt.show()

classifier = GaussianNB()
Y = df['sex']
Y = np.array(Y)
X = df[['actual', 'ideal', 'diff']]
X = np.array(X)
classifier.fit(X,Y)

print classifier.predict([[145, 160, -15]])
#Male

print classifier.predict([[160, 145, 15]])
#Female

