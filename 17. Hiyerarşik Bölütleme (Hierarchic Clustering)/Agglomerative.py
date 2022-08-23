import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_ac = ac.fit_predict(X)

plt.figure(1)
plt.scatter(X[y_pred_ac==0,0], X[y_pred_ac==0,1], s=100, c='red')
plt.scatter(X[y_pred_ac==1,0], X[y_pred_ac==1,1], s=100, c='blue')
plt.scatter(X[y_pred_ac==2,0], X[y_pred_ac==2,1], s=100, c='green')
plt.scatter(X[y_pred_ac==3,0], X[y_pred_ac==3,1], s=100, c='orange')
plt.title('HC')
plt.show()

import scipy.cluster.hierarchy as sch
plt.figure(2)
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()
# 2'ye ayırmak en mantıklısı 3'e ayırmak mantıksız biz 4'e böldük verileri.