## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
~~~
![1](https://github.com/user-attachments/assets/af5eb845-dcbf-4586-a969-49f0c1432ee5)
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![2](https://github.com/user-attachments/assets/e354246c-628a-4919-a3b9-5c72b1ba536b)
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![3](https://github.com/user-attachments/assets/04f495c3-5d44-4017-b218-82a08eed910f)
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~
![4](https://github.com/user-attachments/assets/c24bc60c-8197-40e6-9d2e-dc30481deb82)
~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
~~~
![5](https://github.com/user-attachments/assets/449ddff7-c35d-4a5f-8ddf-0d47f7093efc)
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
![6](https://github.com/user-attachments/assets/8509fb8a-6a23-4327-8e67-b391951ff5ad)
~~~
pip install --upgrade category_encoders
~~~
![7](https://github.com/user-attachments/assets/2a96b374-edf4-4dc1-98d5-0ce88cc2d121)
~~~
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df=pd.read_csv("/content/data.csv")
df
~~~
![8](https://github.com/user-attachments/assets/ce4a6146-02f6-46ef-9032-e017db1f81b0)
~~~
be= BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb1
~~~
![9](https://github.com/user-attachments/assets/3e761374-d85c-4566-9779-465f2bee6024)
~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
~~~
![10](https://github.com/user-attachments/assets/27666bc0-ee65-46cf-b2a4-bfca2c41d70a)
~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
~~~
![11](https://github.com/user-attachments/assets/ad6a1ed4-1abc-4d6f-aaad-7f0574c96e83)
~~~
df.skew()
~~~
![12](https://github.com/user-attachments/assets/f53f64ba-b897-4714-819d-a2d1b2b15400)
~~~
np.log(df["Highly Positive Skew"])
~~~
![13](https://github.com/user-attachments/assets/366e556a-9819-4d82-b109-6849a756abf4)
~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~
![14](https://github.com/user-attachments/assets/2dbff0e9-3bdc-48b2-a0bb-90469ccf32dc)
~~~
np.sqrt(df['Highly Positive Skew'])
~~~
![15](https://github.com/user-attachments/assets/33297404-3e1d-41fc-85f4-9f73770d983b)
~~~
np.square(df["Highly Positive Skew"])
~~~
![16](https://github.com/user-attachments/assets/99fb77a2-a05c-4064-8a9d-e742d7aa4717)
~~~
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df['Highly Positive Skew'])
df
~~~
![17](https://github.com/user-attachments/assets/26bf42b4-e823-4202-85e8-8054182b94d9)
~~~
df.skew()
~~~
![18](https://github.com/user-attachments/assets/f4f53c67-5b1f-4985-bf52-ffe382176e72)
~~~
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
~~~
![19](https://github.com/user-attachments/assets/d949d323-4fde-4e96-8699-d650872426b1)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~
![20](https://github.com/user-attachments/assets/a9b52afb-3358-4d9f-b5ad-2458dc8b8d8c)
~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![21](https://github.com/user-attachments/assets/d63500ca-c6da-49ad-8ab2-2a91e59c16b9)
~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
![22](https://github.com/user-attachments/assets/3711594c-5e29-4069-85fb-7c591a9fc521)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![23](https://github.com/user-attachments/assets/9c0c506f-dc1e-4457-89e5-96ffceb8417f)
~~~
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~
![24](https://github.com/user-attachments/assets/ff0ceedf-73c7-4db2-99a8-8d0b677c915a)
~~~
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
~~~
![25](https://github.com/user-attachments/assets/542f60bc-0585-4032-a6bf-981d58c98255)
~~~
dt=pd.read_csv("/content/titanic_dataset (2).csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
~~~
![26](https://github.com/user-attachments/assets/7c992c19-5983-4041-a5ee-12679b164b14)
~~~
sm.qqplot(dt['Age_1'],line='45')
plt.show()
~~~
![27](https://github.com/user-attachments/assets/0e0e9895-8f63-4f52-b64e-f81e4c185bda)
# RESULT:
  Thus we performed Feature Encoding and Transformation process

       
