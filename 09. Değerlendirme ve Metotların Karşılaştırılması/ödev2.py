# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2.1. veri yükleme
veriler = pd.read_csv("maaslar_yeni.csv")
print(veriler)

#♠ data frame dilimleme (slice)
x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]

# NumPY dizi (array) dönüşümü
X = x.values
Y = y.values

# Linear Regression
# Doğrusal Model Oluşturma

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x,y)

import statsmodels.api as sm
model = sm.OLS(lr.predict(X), Y)
print(model.fit().summary())


from sklearn.metrics import r2_score

print("===Linear R2 Değeri===")
print(r2_score(y, lr.predict(x)))

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
print(lr.predict([[10,10,100]]))


print(lr2.predict(pr2.fit_transform([[10,10,100]])))


print("===Polynomial R2 Değeri===")
print(r2_score(Y, lr3.predict(pr3.fit_transform(X))))

# Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

# Support Vetor Machine
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_olcekli, y_olcekli)
"""
plt.scatter(x_olcekli, y_olcekli)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli))
plt.show()
"""
print(svr_reg.predict([[10,10,100]]))

print("===Support Vector R2 Değeri===")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
"""
plt.scatter(X, Y, color="red")
plt.plot(X, r_dt.predict(X), color="blue")
plt.show()
"""

print(r_dt.predict([[10,10,100]]))
print("===Decision Tree R2 Değeri===")
print(r2_score(Y, r_dt.predict(X)))


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
# ilk parametre kaç tane decision tree çizeceğini belirtiyor.
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[10,10,100]]))
"""
plt.scatter(X, Y, color="red")
plt.plot(X, rf_reg.predict(X), color="blue")
"""

print("===Random Forest R2 Değeri===")
print(r2_score(Y, rf_reg.predict(X)))



# Özet R2 Değerleri

print("\n"*5)
print("="*50)

print("===Linear R2 Değeri===".center(50))
print(r2_score(y, lr.predict(x)))
print("CEO: ", lr.predict([[10,10,100]]))
print("Müdür: ", lr.predict([[7,10,100]]))

print("="*50)

print("===Polynomial R2 Değeri===".center(50))
print(r2_score(Y, lr3.predict(pr3.fit_transform(X))))
print("CEO: ", lr2.predict(pr2.fit_transform([[10,10,100]])))
print("Müdür: ", lr2.predict(pr2.fit_transform([[7,10,100]])))

print("="*50)

print("===Support Vector R2 Değeri===".center(50))
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))
print("CEO: ", svr_reg.predict([[10,10,100]]))
print("Müdür: ", svr_reg.predict([[7,10,100]]))

print("="*50)

print("===Decision Tree R2 Değeri===".center(50))
print(r2_score(Y, r_dt.predict(X)))
print("CEO: ", r_dt.predict([[10,10,100]]))
print("Müdür: ", r_dt.predict([[7,10,100]]))

print("="*50)

print("===Random Forest R2 Değeri===".center(50))
print(r2_score(Y, rf_reg.predict(X)))
print("CEO: ", rf_reg.predict([[10,10,100]]))
print("Müdür: ", rf_reg.predict([[7,10,100]]))

print("="*50)
print("\n"*5)




import statsmodels.api as sm
model = sm.OLS(lr.predict(X), X)
print(model.fit().summary())

# Başka bir şey
X = np.append(arr = np.ones((30,1)).astype(int), values=veriler.iloc[:,:-2], axis=1 )
X_l = veriler.iloc[:,[2,3,4]].values
r_ols = sm.OLS(endog = veriler.iloc[:,-2:-1], exog =X_l.astype(float))
r = r_ols.fit()
print(r.summary())


print(veriler.corr())





















