# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2.1. veri yükleme
veriler = pd.read_csv("maaslar.csv")
print(veriler)

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

# Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)

plt.scatter(x,y)
plt.plot(x,lr.predict(x))
plt.show()


# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 2)
x_poly = pr.fit_transform(X)
print(x_poly)
lr2 = LinearRegression()
lr2.fit(x_poly, y)
plt.scatter(X, Y)
plt.plot(X, lr2.predict(pr.fit_transform(X)))
plt.show()


# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 4)
x_poly = pr.fit_transform(X)
print(x_poly)
lr2 = LinearRegression()
lr2.fit(x_poly, y)
plt.scatter(X, Y)
plt.plot(X, lr2.predict(pr.fit_transform(X)))
plt.show()

print(lr.predict([[6.6]]))
print(lr.predict([[11]]))

print(lr2.predict(pr.fit_transform([[6.6]])))
print(lr2.predict(pr.fit_transform([[11]])))
