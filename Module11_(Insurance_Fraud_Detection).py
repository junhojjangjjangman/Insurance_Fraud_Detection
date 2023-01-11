import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

df = pd.read_csv("C:/Users/15/Desktop/DataSet/[Dataset]_Module11_(Insurance).csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.info())
print(df.nunique())
print(df.describe().T)

colors = ['y', 'dodgerblue']
x = np.arange(2)
values = df.groupby('fraud_reported')['fraud_reported'].count()
x_label = np.unique(df['fraud_reported'])

plt.bar(x, values, color=colors)
plt.xticks(x, x_label)
plt.title("fraud_reported");
plt.ylabel("Count");
plt.xlabel('fraud_reported')
plt.show()

print((df.groupby('fraud_reported')['fraud_reported'].count()))
print(np.unique(df['fraud_reported']))

ratio = df.groupby('fraud_reported')['fraud_reported'].count()
labels = np.unique(df['fraud_reported'])

plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=90, counterclock=False)
plt.legend()
plt.show()


# pandas의 corr() 함수를 사용하여 상관 행렬 가져오기
corr_matrix = df.corr()

fig = go.Figure(data = go.Heatmap(
                                z = corr_matrix.values,
                                x = list(corr_matrix.columns),
                                y = list(corr_matrix.index)))

fig.update_layout(title = 'Correlation_Insurance_Fraud')

# print("z: ",corr_matrix.values,"\nx: ",list(corr_matrix.columns),"\ny:", list(corr_matrix.index))

fig.show()

# Attrition_rate는 예측할 레이블 또는 출력입니다.
# features는 Attrition_rate를 예측하는 데 사용됩니다.
label = ["fraud_reported"]
features = df.columns.values.tolist()
features_remove_fraud_reported = df.columns.values.tolist()
features_remove_fraud_reported.remove('fraud_reported')

print(features_remove_fraud_reported)
featured_data = df.loc[:,features]
print(featured_data.shape)

fraud_reported_data = featured_data.dropna(axis=0)
print(fraud_reported_data.shape)

# fraud_reported를 Target으로 Dataset 나누기
X = fraud_reported_data.loc[:,features_remove_fraud_reported]
y = fraud_reported_data.loc[:,label]
print(X.shape)
print(y.shape)

sc = StandardScaler()#정규화 하는 것
X = sc.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=34,test_size=0.2)
print("x_train values count:",len(X_train))
print("y_train values count:",len(y_train))
print("x_test values count:",len(X_test))
print("y_test values count:",len(y_test))

rfc = RandomForestClassifier(n_estimators= 200, max_features = 15, max_depth = 15, random_state = 1)
rfc.fit(X_train, y_train)
preds = rfc.predict(X_test)

score = accuracy_score(y_test,preds)*100
print(score)
print("훈련 세트 정확도: {:.3f}".format(rfc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(rfc.score(X_test, y_test)))
print(classification_report(y_test, preds))

print("auc score: ", roc_auc_score(y_test, preds))

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
f, ax = plt.subplots(figsize=(10, 10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()# 한 번 이해해 보기 나중에

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

gb_score = accuracy_score(y_test,gb_preds)*100
print(gb_score)
print("훈련 세트 정확도: {:.3f}".format(gb.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gb.score(X_test, y_test)))
print(classification_report(y_test, gb_preds))

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

xgb_score = accuracy_score(y_test, xgb_preds)*100
print(xgb_score)
print("훈련 세트 정확도: {:.3f}".format(xgb.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(xgb.score(X_test, y_test)))
print(classification_report(y_test, xgb_preds))


