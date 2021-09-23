import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV


dataset = pd.read_csv('flight_delay.csv')
dataset.head()
dataset.describe()


#encodinng data
label_enc = preprocessing.LabelEncoder()
dataset['Depature Airport'] = label_enc.fit_transform(dataset['Depature Airport'])
dataset['Destination Airport'] = label_enc.fit_transform(dataset['Destination Airport'])
dataset.head()


#checking for missing values
dataset.isnull()
dataset.isnull().sum()


dataset['Scheduled depature time'] = pd.to_datetime(dataset['Scheduled depature time'])
dataset['Scheduled arrival time'] = pd.to_datetime(dataset['Scheduled arrival time'])

#duration
duration = dataset['Scheduled arrival time'] - dataset['Scheduled depature time']
flight_dur_sec = duration.dt.total_seconds()
flight_dur = flight_dur_sec/60
dataset['Flight duration'] = flight_dur

dataset


#splitting data into train and test data

train_data = dataset.drop(dataset[pd.DatetimeIndex(dataset['Scheduled depature time']).year == 2018].index)
test_data = dataset.drop(dataset[pd.DatetimeIndex(dataset['Scheduled depature time']).year < 2018].index)


corr_mat = train_data.corr()
sb.heatmap(corr_mat, annot = True)
plt.show()


#visualising data

plt.plot(dataset['Flight duration'], dataset['Delay'], 'bo')
plt.title('Flight duration vs Delay')
plt.xlabel('Flight duration (min)')
plt.ylabel('Delay (min)')
plt.show()


#define the outlier and remove

sb.boxplot(data = dataset['Delay'], x = dataset['Flight duration'])
plt.title('Boxplot for outliers detection')


#outliers removing
train_data = train_data.drop(train_data[(train_data['Flight duration'] > 900)].index)


#obtaining X_train, y_train, x_test, y_test

X_train = train_data.drop(['Scheduled depature time', 'Scheduled arrival time','Delay'], axis=1)
y_train = train_data['Delay']
x_test = test_data.drop(['Scheduled depature time', 'Scheduled arrival time','Delay'], axis=1)
y_test = test_data['Delay']


#Linear Regression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#checking train data
y_train_pred = regressor.predict(X_train)

#checking test data
y_test_pred =  regressor.predict(x_test)


#checking train data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_train_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_train_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('R score', metrics.r2_score(y_train, y_train_pred))


#checking test data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('R score', metrics.r2_score(y_test, y_test_pred))


#Polynomial Regression

poly_reg = PolynomialFeatures(degree=3)
x_train_poly = poly_reg.fit_transform(X_train)
poly_reg_model = LinearRegression()
poly_reg_model.fit(x_train_poly,y_train)

#checking train data
y_train_pred = poly_reg_model.predict(x_train_poly)

#checking test data
y_test_pred = poly_reg_model.predict(poly_reg.fit_transform(x_test))


#checking train data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_train_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_train_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('R score', metrics.r2_score(y_train, y_train_pred))


#checking test data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('R score', metrics.r2_score(y_test, y_test_pred))



#Lasso

X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=123)

alphas = [2.2, 2, 1.5, 1.3, 1.2, 1.1, 1, 0.3, 0.1]
losses = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)
    losses.append(mse)
plt.plot(alphas, losses)
plt.title("Lasso alpha value selection")
plt.xlabel("alpha")
plt.ylabel("Mean squared error")
plt.show()

best_alpha = alphas[np.argmin(losses)]
print("Best value of alpha:", best_alpha)



lasso = Lasso(best_alpha)
lasso.fit(X_train, y_train)

#checking train data
y_train_pred = lasso.predict(X_train)

#checking test data
y_test_pred = lasso.predict(x_test)


#checking train data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_train_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_train_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('R score', metrics.r2_score(y_train, y_train_pred))


#checking test data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print('R score', metrics.r2_score(y_test, y_test_pred))
