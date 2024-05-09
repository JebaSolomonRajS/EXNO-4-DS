# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/8590ae3f-98ea-46ea-b6a1-c36695da003d)

```
df.dropna()
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/1c8e579c-a5a8-4f3e-a187-b0046e9dff67)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/43cc40c2-f7dc-45d8-b15a-e5d1c6760bf8)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/69bf430f-8aca-43d3-beb5-782879087091)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/f03d1b12-54c9-450e-9c3e-3cde72d949fb)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/0d468ba9-9103-4f59-8330-58eb3bc12c4d)
```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/e7ca2022-601f-4e73-a47a-3b2ca076e3eb)
```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/4b6b7fdd-1565-40fe-9f83-a49eeda535b5)

```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/6a0cfe0f-07c9-4062-84fc-39b1fc4d2664)

```
data.isnull().sum()
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/dce2b7fc-184c-49e7-882b-7b2f4bef5c49)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/bfea4c4e-d132-49b0-b4f4-639d0098ec98)
```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/8c195fd1-307c-47a8-b6a3-5edcf5b3f9d5)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/026c94a5-b1ce-4ff6-b3a8-a75a85c8dd86)
```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/64c81a1e-7d72-41b7-a249-7ce17101b8ec)
```
data2
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/23e38208-55cf-459b-a578-0cc0b2b429c2)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/201328bb-ebb7-42e6-91c1-cffbf9320a98)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/b36a33bd-d002-4227-a7fc-48618cf009ef)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/17f03070-6eda-43d8-9a6d-24131140b866)
```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]

```
x = new_data[features].values
print(x)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/34be9fa3-5301-4b07-ab8a-1fd379d47106)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/caaf5f68-984f-4653-b4d3-462d0f0e0812)
```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/2dcf7c84-7ce0-429a-936f-5809d3bb9ba5)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
0.8392087523483258

```

print('Misclassified samples: %d' % (test_y != prediction).sum())
```
Misclassified samples: 1455

```
data.shape

(31978, 13)
```
## FEATURE SELECTION TECHNIQUES
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/31eeae06-5e92-4bbe-aa81-6d3d13c60f68)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/57d88aaf-dd81-4543-8819-f9a1410bba15)

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/f75a6cb9-8c69-490a-9466-8c3512e0ab68)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/JebaSolomonRajS/EXNO-4-DS/assets/139432449/a45e8e3a-b2f1-41bf-953b-61ce37f297b4)


# RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.

