import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import statsmodels.api as smf
from sklearn import cross_validation
from sklearn.cross_validation import KFold

df = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')

df['Interest.Rate'] = df['Interest.Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
df['Loan.Length'] = df['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
df['FICO.Score'] = df['FICO.Range'].map(lambda x: int(x[:3]))

kf = KFold(2500, n_folds=10)

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()
	

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	print("%s %s" % (train, test))
	cv = df.iloc[train]
	fico = cv['FICO.Score']
	loanamt = cv['Amount.Requested']
	intrate = cv['Interest.Rate']
	y = np.matrix(intrate).transpose()
	x1 = np.matrix(fico).transpose()
	x2 = np.matrix(loanamt).transpose()
	x = np.column_stack([x1, x2])
	X = smf.add_constant(x)
	model = smf.OLS(y, X)
	f = model.fit()
	f.summary()
#R squared average = .65

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()

for train, test in kf:
	cv_test = df.iloc[test]
	intrate_test = cv_test['Interest.Rate']
	loanamt_test = cv_test['Amount.Requested']
	fico_test = cv_test['FICO.Score']
	y_test = np.matrix(intrate_test).transpose()
	x1_test = np.matrix(fico_test).transpose()
	x2_test = np.matrix(loanamt_test).transpose()
	x = np.column_stack([x1_test, x2_test])
	model = smf.OLS(y_test, X)
	f = model.fit()
	f.summary()
#R squared average = 0.71
