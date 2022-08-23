# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv', header = None)

# En uzun alışveriş listesini bilmediğimiz için her alışverişte 20 kalem
# alındığını varsayarak litemizi hazırlıyoruz.
t = []
for i in range(1,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])

# 20 kalemlik listelerde oldukça fazla nan değeri olduğu için onları temizliyoruz.
new_t = []
for liste in t:
    new_t.append([x for x in liste if pd.isnull(x) == False and x != 'nan'])


from apyori import apriori
kurallar = apriori(new_t, min_support=0.01, min_confidence=0.2, min_lift=2.5, min_length=2)
print(list(kurallar))