import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pylab as pl
import re
import numpy as np

subjects = pd.read_csv("/Users/rohanpatel/Downloads/UCI HAR Dataset/train/subject_train.txt", header=None, delim_whitespace=True, index_col=False)
subjects.columns = ['Subject']
#21 total participants and 7352 total observations

features = pd.read_csv("/Users/rohanpatel/Downloads/UCI HAR Dataset/features.txt", header=None, delim_whitespace=True, index_col=False)
#561 features

x_variables = pd.read_csv("/Users/rohanpatel/Downloads/UCI HAR Dataset/train/X_train.txt", header=None, delim_whitespace=True, index_col=False)

fix = []; fix2 = []; fix3 = []; fix4 = []; fix5 = []; fix6 = []
for el in features[1]:
	fix.append(re.sub('[()-]', '', el))
for el in fix:
	fix2.append(re.sub('[,]', '_', el))
for el in fix2:
	fix3.append(el.replace('Body', ''))
for el in fix3:
	fix4.append(el.replace('Mag', ''))
for el in fix4:
	fix5.append(el.replace('mean', 'Mean'))
for el in fix5:
	fix6.append(el.replace('std', 'STD'))

x_variables.columns = fix6
y_variable = pd.read_csv("/Users/rohanpatel/Downloads/UCI HAR Dataset/train/y_train.txt", header=None, delim_whitespace=True, index_col=False)
y_variable.columns = ['Activity']

data = pd.merge(y_variable, x_variables, left_index=True, right_index=True)
data = pd.merge(data, subjects, left_index=True, right_index=True)

data['Activity'] = pd.Categorical(data['Activity']).labels


#divide data into training, testing, and validation subjects
train = data.query('Subject >= 27')
test = data.query('Subject <= 6')
validation = data.query("(Subject >= 21) & (Subject < 27)")

#fitting the random forest model
train_target = train['Activity']
train_data = train.ix[:,1:-2]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc.fit(train_data, train_target)

#oob score shows accuracy of the model
rfc.oob_score

#important features
important = rfc.feature_importances_
indices = np.argsort(important)[::-1]
for i in range(10):
    print("%d. feature %d (%f)" % (i + 1, indices[i], important[indices[i]]))

#define validation sets for predictions
validation_target = validation['Activity']
validation_data = validation.ix[:1,-2]
validation_pred = rfc.predict(validation_data)

#define target sets for predictions
test_target = test['Activity']
test_data = test.ix[:,1:-2]
test_pred = rfc.predict(test_data)

#accuracy scores
print("mean accuracy score for validation set is %f" %(rfc.score(validation_data, validation_target)))
print("mean accuracy score for test set is %f" %(rfc.score(test_data, test_target)))

#confusion matrix
test_confusion_matrix = skm.confusion_matrix(test_target, test_pred)
pl.matshow(test_confusion_matrix)
pl.title('Test Data Confusion Matrix')
pl.colorbar()
pl.show()


#accuracy, precision, recall, and F1 score
print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))
print("Precision = %f" %(skm.precision_score(test_target, test_pred)))
print("Recall = %f" %(skm.recall_score(test_target, test_pred)))
print("F1 score = %f" %(skm.f1_score(test_target, test_pred)))


