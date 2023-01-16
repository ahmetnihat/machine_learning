X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())