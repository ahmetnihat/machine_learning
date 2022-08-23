# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2.1. veri yükleme
veriler = pd.read_csv("maaslar.csv")
print(veriler)

#♠ data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# NumPY dizi (array) dönüşümü
X = x.values
Y = y.values

# Linear Regression
# Doğrusal Model Oluşturma

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)


# Polynomial Regression
# Doğrusal Olmayan (nonlinear model)
# 2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
pr2 = PolynomialFeatures(degree = 2)
x_poly = pr2.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(x_poly, y)

# 4. dereceden polinom
pr3 = PolynomialFeatures(degree = 4)
x_poly3 = pr3.fit_transform(X)
lr3 = LinearRegression()
lr3.fit(x_poly3, y)

"""
# Görselleştirme
plt.scatter(x,y)
plt.plot(x,lr.predict(x))
plt.show()

plt.scatter(X, Y)
plt.plot(X, lr2.predict(pr2.fit_transform(X)))
plt.show()

plt.scatter(X, Y)
plt.plot(X, lr3.predict(pr3.fit_transform(X)))
plt.show()
"""

# Tahminler
print(lr.predict([[6.6]]))
print(lr.predict([[11]]))

print(lr2.predict(pr2.fit_transform([[6.6]])))
print(lr2.predict(pr2.fit_transform([[11]])))

# Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)


from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_olcekli, y_olcekli)
"""
plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli))
plt.show()
"""
print(svr_reg.predict([[6.6]]))
print(svr_reg.predict([[11]]))



from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor()
r_dt.fit(X,Y)

plt.scatter(X, Y, color="red")
plt.plot(X, r_dt.predict(X), color="blue")


print(r_dt.predict([[6.6]]))
print(r_dt.predict([[11]]))










