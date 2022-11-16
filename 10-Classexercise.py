from statistics import mode
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise
from sklearn.metrics import auc, roc_curve, confusion_matrix



# 2 Read AirBnB data
df = pd.read_csv("../data/AB_NYC_2019.csv")

# 3 Add is_cheap column based on median values
df['is_cheap'] = df['price'] < df['price'].median()
print(df['is_cheap'])
print(df['price'])

# 4 Create classifier model from KNN
model = KNeighborsClassifier(n_neighbors=10)

# 5 
print(df.columns)
latlong = pd.DataFrame()
latlong['longitude'] = df['longitude']
latlong['latitude'] = df['latitude']

print(latlong.head())
print(latlong.shape)
print(df.shape)

# 6
X = latlong
y = df['is_cheap']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 7
model.fit(X_train, y_train)

# 8
y_pred = model.predict(X_test)

# 9
fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=True)
print(fpr)
print(tpr)
print(threshold)
print(auc(fpr,tpr))



# 10
## 10.1
OHEdf = pd.get_dummies(df, columns=['neighbourhood', 'neighbourhood_group', 'room_type'])
print(X_train.columns)

## 10.2
X_train, X_test, y_train, y_test = train_test_split(OHEdf, y, test_size=0.33, train_size=0.67)
indepedent_variables = [col for col in OHEdf if col.startswith('neighbourhood') or col.startswith('room_type')]
indepedent_variables.append('longitude')
indepedent_variables.append('latitude')
indepedent_variables.append('number_of_reviews')
indepedent_variables.append('reviews_per_month')

## 10.3
standard_scaler_model = StandardScaler().fit(X_train[indepedent_variables])

## 10.4
X_train_norm = np.nan_to_num(standard_scaler_model.transform(X_train[indepedent_variables]))
print(X_train_norm)

## 10.5
X_test_norm = np.nan_to_num(standard_scaler_model.transform(X_test[indepedent_variables]))
print(X_test_norm)

## 10.6
KModel = KNeighborsClassifier(n_neighbors=10)
KModel.fit(X_train_norm, y_train)

## 10.7
y_pred = KModel.predict(X_test_norm)
fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=True)
print(auc(fpr,tpr))

## 10.8
metrics = ['minkowski','manhattan']

def measureDistance(k: int, dist: str) -> float:
    model = KNeighborsClassifier(n_neighbors=k, metric=dist)

    model.fit(X_train_norm, y_train)
    y_pred = KModel.predict(X_test_norm)
    fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=True)
    return print(dist,'=', auc(fpr,tpr))

## 10.9
#cosine = pairwise_distances(X=2, metric='cosine')
#print(cosine)
print(X_test_norm.shape)
md = pairwise.manhattan_distances(X_test_norm)
print(md.shape)
print(md)
measureDistance(2,'minkowski')
measureDistance(2,'manhattan')


