# 1. kütüphaneler
import pandas as pd


url = "https://bilkav.com/satislar.csv"

veriler = pd.read_csv(url).values

X = veriler[:, 0:1]
Y = veriler[:, 1]

bolme = 0.33

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=bolme)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)
predict_fit = lr.predict(x_test)

import pickle

dosya = "model.kayit"

pickle.dump(lr, open(dosya, 'wb'))

yuklenen = pickle.load(open(dosya, 'rb'))
predict_load = yuklenen.predict(x_test)