# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2. veri ön işleme

# 2.1. veri yükleme
veriler = pd.read_csv("veriler.csv")
print(veriler)

x = veriler.iloc[:,1:4].values # bağımsız değişkenler
y = veriler.iloc[:,4:].values # bağımlı değişkenler


# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

x_test[-1,:] = np.array([180, 80, 32])
# Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
# 1. sinde fit yazarak x_train verilerine göre öğrenme yaparak scale etti
# 2. satırda yazarsak x_test için ayrı öğrenme yaparak scale edecek

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)