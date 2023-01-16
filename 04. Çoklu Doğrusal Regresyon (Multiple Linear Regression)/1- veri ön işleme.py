# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2. veri ön işleme

# 2.1. veri yükleme
veriler = pd.read_csv("veriler.csv")
print(veriler)

# eksik veriler

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)


# verileri kategorikleştirme
# Encoder: Nominal Ordinal (kategorik) -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


# verileri kategorikleştirme
# Encoder: Nominal Ordinal (kategorik) -> Numeric
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

# numpy dizileri dataframe dönüşümü
sonuc = pd.DataFrame(data=ulke, index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=c[:,:1], index=range(22), columns=["cinsiyet"])
print(sonuc3)

# df'leri birleştirme
s = pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)







