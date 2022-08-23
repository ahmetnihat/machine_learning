# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2. veri ön işleme

# 2.1. veri yükleme
veriler = pd.read_csv("Wine.csv")
print(veriler)

X = veriler.iloc[:, 0:13].values
Y = veriler.iloc[:, 13].values


# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# PCA dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# PCA dönüşümünden sonra gelen LR
classifer2 = LogisticRegression(random_state=0)
classifer2.fit(X_train2, y_train)

# Tahminler
y_pred = classifier.predict(X_test)
y_pred2 = classifer2.predict(X_test2)

from sklearn.metrics import confusion_matrix
# actual / PCA olmadan çıkan sonuç
cm = confusion_matrix(y_test, y_pred)
# actual / PCA sonrası çıkan sonuç
cm2 = confusion_matrix(y_test, y_pred2)
# PCA sonrası / PCA öncesi çıkan sonuç
cm3 = confusion_matrix(y_pred, y_pred2)

# LDA için sınıflar önemlidir PCA için önemsizdir.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# LDA dönüşümünden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)

# LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)
# LDA sonra / orijinal
cm_lda = confusion_matrix(y_test, y_pred_lda)























