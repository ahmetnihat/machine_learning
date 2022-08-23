import numpy as np
import pandas as pd

yorumlar = pd.read_csv('Restaurant_Reviews.csv', error_bad_lines=False)
yorumlar.dropna(subset = ["Liked"], inplace=True)
yorumlar = yorumlar.reset_index(drop=True, inplace=False)
#yorumlar.drop(['index'], axis=1)
print(yorumlar.tail(5))

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

durma = nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing (Önişleme)
derlem = []
for i in range(704):
    yorum =re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    # yorum listesinin içindeki kelimeleri dolaşarak stopword olup olmadığına bakacak
    # eğer stopwords değilse o kelimeyi stem edecek yani kökünü bulacak ve yeni
    # yorumumuz olarak kaydedecek.
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
# Feature Extraction (Öznitelik Çıkarımı)
# Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değiken
y = yorumlar.iloc[:,1] # bağımlı değişken

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
"""
np.any(np.isnan(X_train))
np.any(np.isnan(y_train))
np.isfinite(X_train.all())
np.isfinite(y_train.all())
"""
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

toplam_veri = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
doğrular = cm[0][0] + cm[1][1]
doğruluk_oranı = (doğrular / toplam_veri) * 100