from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, DBSCAN, Birch, MeanShift
import numpy as np

### Part 1

X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

print(X)
print(y)

sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)
plt.show()

model = AffinityPropagation(damping=0.7)
print(model)

model.fit(X)
print(model)

p = model.predict(X)
print(p)

uniques = np.unique(p)
print(uniques)


sns.scatterplot(x=X[:,0], y=X[:,1],hue=p)

plt.show()

dbscan_model = DBSCAN(eps=0.5)
db = dbscan_model.fit_predict(X)
sns.scatterplot(x = X[:,0], y = X[:,1], hue=db)
plt.show()

meanshift_model = MeanShift()
ms = meanshift_model.fit_predict(X)
sns.scatterplot(x = X[:,0], y = X[:,1], hue=ms)
plt.show()

birch_model = Birch()
bm = birch_model.fit_predict(X)
sns.scatterplot(x = X[:,0], y = X[:,1], hue=bm)
plt.show()