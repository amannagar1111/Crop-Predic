

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Crop_recommendation.csv")



df.head()

df.shape

df.info()

df.describe()

df['label'].unique()

plt.figure(figsize=(15,6))
sns.barplot(x='label',y='N',data = df,palette='hls')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15,6))
sns.barplot(x='label',y='P',data = df,palette='hls')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15,6))
sns.barplot(x='label',y='K',data = df,palette='hls')
plt.xticks(rotation=90)
plt.show()

sns.distplot(df['N'])

sns.distplot(df['P'])

df.isna().sum()

df[df['N'] == 0].sum()

"""**Data preproccessing**"""

from sklearn.model_selection import train_test_split
X = df.drop('label',axis=1)
y = df['label']

X.head()

y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train

from sklearn.preprocessing import StandardScaler,MinMaxScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""**Naive Bayes**"""

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score,classification_report
gb = GaussianNB()
mb = MultinomialNB()
bb = BernoulliNB()

gb.fit(X_train,y_train)
y_predict = gb.predict(X_test)
print("Accuracy score :",accuracy_score(y_test,y_predict))
print("Classification Report : \n",classification_report(np.array(y_test),y_predict))

mb.fit(X_train_scaled,y_train)
y_predict = mb.predict(X_test_scaled)
print(accuracy_score(y_test,y_predict))

bb.fit(X_train_scaled,y_train)
y_predict = bb.predict(X_test_scaled)
print(accuracy_score(y_test,y_predict))

"""**Logistic Regression**"""

standard_scaler = StandardScaler()
X_train_sc = standard_scaler.fit_transform(X_train)
X_test_sc = standard_scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,y_train)

y_predict = lr.predict(X_test)

print(accuracy_score(y_test,y_predict))



"""**KNN**"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_predict = knn.predict(X_test)

accuracy_score(y_test,y_predict)



"""Predict"""

testdf = pd.DataFrame({'N':[50],'P':[20],'K':[43],'temperature':[20.87],'humidity':[82],'ph':[6.5],'rainfall':[202.93]})

testdf

le.inverse_transform(gb.predict(testdf))

