# 1. kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# kod bölümü

# 2. veri ön işleme

# 2.1. veri yükleme
veriler = pd.read_csv("Churn_Modelling.csv")
print(veriler)

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

# verileri kategorikleştirme
# Encoder: Nominal Ordinal (kategorik) -> Numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencode_X_1 = LabelEncoder()
X[:, 1] = labelencode_X_1.fit_transform(X[:, 1])

labelencode_X_2 = LabelEncoder()
X[:, 2] = labelencode_X_2.fit_transform(X[:, 2])

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float), [1])],
                        remainder="passthrough")

X = ohe.fit_transform(X)
X = X[:,1:]

# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.21, random_state=0)


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=classifier.classes_)
disp.plot()















