# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2. veri ön işleme

# 2.1. veri yükleme
veriler = pd.read_csv("Social_Network_Ads.csv")
print(veriler)

x = veriler.iloc[:, [2, 3]].values # bağımsız değişkenler
y = veriler.iloc[:, 4].values # bağımlı değişkenler


# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
# 1. sinde fit yazarak x_train verilerine göre öğrenme yaparak scale etti
# 2. satırda yazarsak x_test için ayrı öğrenme yaparak scale edecek


from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
y_pred_svc = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test, y_pred_svc)

from sklearn.model_selection import cross_val_score
"""
1. estimator : classifier (bizim durum)
2. X
3. Y
4. cv : kaç katlamalı
"""
başarı = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=4)
başarı.mean()
başarı.std()

# parametre optimizasyonu ve algoritma seçimi
from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5], 'kernel':['linear']},
     {'C':[1,10,100,1000], 'kernel':['rbf'],
      'gamma': [1,0.5,0.1,0.01,0.001]} ]

"""
GSCV parametreleri
1. estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
2. param_grid : parametreler/denenecekler
3. scoring : neye göre skorlanacak : örn : accuracy
4. cv : kaç katlamalı olacağı
n_jobs : aynı anda çalışacak iş
"""

gs = GridSearchCV(estimator = classifier, # SVM algoritması
                  param_grid = p,
                  scoring= 'accuracy',
                  cv = 10,
                  n_jobs = -1)

grid_search = gs.fit(X_train, y_train)
en_iyi_sonuç = grid_search.best_score_
en_iyi_parametreler = grid_search.best_params_




























