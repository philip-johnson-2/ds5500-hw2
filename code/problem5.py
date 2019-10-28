'''
Problem 5:

Model GDP vs child mortality over time
'''

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# assign github URLs
gdp_url = 'https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/ddf--datapoints--gdppercapita_us_inflation_adjusted--by--geo--time.csv'
infant_mortaility_url = 'https://raw.githubusercontent.com/open-numbers/ddf--gapminder--systema_globalis/master/ddf--datapoints--infant_mortality_rate_per_1000_births--by--geo--time.csv'

# read dataset
gdp_df = pd.read_csv(gdp_url, skiprows=1, names=['Geo','Year','gdp_per_capita'])
infant_mortality_df = pd.read_csv(infant_mortaility_url, skiprows=1, names=['Geo','Year','mortality_rate'])

# merge data by year to ensure consistency of data
df = pd.merge(gdp_df,
            infant_mortality_df[['Geo','Year','mortality_rate']],
            left_on = ['Geo','Year'], 
            right_on = ['Geo','Year'],
            how = 'left')

# remove NA rows
df = df.dropna()

# simple linear regression

'''
    compare the relationship 
    of year to mortality rate
'''

# define independent and dependent variables
X = df["Year"].values.reshape(-1,1) 
y = df["mortality_rate"].values.reshape(-1,1) 

# split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit model
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

# retrieve the intercept and slope
print(regressor.intercept_)
print(regressor.coef_)

# predict y
y_pred = regressor.predict(X_test)

# plot predicted against actual
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

# print error rates
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# multivariate regression

'''
    mortality rate ~ year + gdp 

    run linear regression for year and 
    gpd to mortality rate to determine 
    relationship between year and gdp 
    to mortality rate
'''

# define independent and dependent variables
X = df[["Year","gdp_per_capita"]] 
y = df["mortality_rate"] 

# split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit model
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

# retrieve coefficient information
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# predict y
y_pred = regressor.predict(X_test)

# print error rates
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


'''
    gdp ~ year + mortality rate

    run linear regression for year and 
    mortality rate to gdp to determine 
    relationship between year and mortality rate 
    to gdp
'''

# define independent and dependent variables
X = df[["Year","mortality_rate"]] 
y = df["gdp_per_capita"] 

# split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit model
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

# retrieve coefficient information
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# predict y
y_pred = regressor.predict(X_test)

# print error rates
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

