# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2. veri ön işleme

# 2.1. veri yükleme
veriler = pd.read_csv("odev_tenis.csv")
print(veriler)

# verileri kategorikleştirme
# Encoder: Nominal Ordinal (kategorik) -> Numeric

from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

# numpy dizileri dataframe dönüşümü
hava_durumu = pd.DataFrame(data=c, index=range(14),columns=["o","r","s"])
son_veriler = pd.concat([hava_durumu, veriler.iloc[:,1:3]], axis=1)
son_veriler = pd.concat([veriler2.iloc[:,-2:], son_veriler], axis=1)
print(son_veriler)

# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(son_veriler.iloc[:,:-1], son_veriler.iloc[:,-1:], test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

X_train = lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=son_veriler.iloc[:,:-1], axis=1 )
X_l = son_veriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = son_veriler.iloc[:,-1:], exog =X_l.astype(float))
r = r_ols.fit()
print(r.summary())

son_veriler = son_veriler.iloc[:,1:]
X = np.append(arr = np.ones((14,1)).astype(int), values=son_veriler.iloc[:,:-1], axis=1 )
X_l = son_veriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = son_veriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

lr.fit(x_train,y_train)


y_pred2 = lr.predict(x_test)






