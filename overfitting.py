import numpy as np
import statsmodels.formula.api as smf

#Set seed for reproducible results
np.random.seed(414)

#Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

#Linear Fit, Training Data
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

#Quadratic Fit, Training Data
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

poly_1.summary()
poly_2.summary()

#Linear Test
poly_1 = smf.ols(formula='y ~ 1 + X', data=test_df).fit()

#Quadratic Test
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=test_df).fit()

poly_1.summary()
poly_2.summary()