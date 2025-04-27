## EXNO-3-DS
## NAME : D KARTHIKEYAN 
## REG_NO : 212224230115
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

```
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/01be7c0b-b31c-4328-aa91-23127d743f32)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/4f5bad10-5f9a-4863-b4b7-2a95b7343a74)

```
df['bo_2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/5f7848d6-ec50-4d4e-9447-86fdf8c63ee7)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/effc446c-2902-4b7c-8b2f-0b0a469a7357)

```
dfc=df.copy()
dfc['con_2']=le.fit_transform(df['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/be55609c-1f50-4aef-8696-7890fff9bf6a)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/868808da-de35-40f3-97af-093556c84a52)

```
!pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/drive/My Drive/data.csv')
df
```
![image](https://github.com/user-attachments/assets/a9b69827-4d83-4dcf-b5ee-cd37cde7980c)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
sfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/158b8197-9ce9-4cb7-862f-f66262981600)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/7fd05845-e7ea-4443-bb39-dbcf6cd54b99)

```
import pandas as pd
from scipy import stats
df=pd.read_csv('/content/drive/My Drive/Data_to_Transform.csv')
df
```
![image](https://github.com/user-attachments/assets/061762c2-7a68-4fac-8ed3-a42631f94876)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/806b34ec-4e68-4950-ab9b-73e87241ca0c)

```
np.reciprocal(df['Moderate Positive Skew'])
```
![image](https://github.com/user-attachments/assets/b0d34b95-83f2-4aa7-82a4-7d8362eccde6)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/46d530c7-6628-474b-9e5e-d999414233db)

```
np.square(df['Highly Negative Skew'])
```
![image](https://github.com/user-attachments/assets/341f08e5-62f3-4fa5-a9f8-c2d4d2614fab)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/84526f8c-6513-4399-872d-434ed6f84ea8)

```
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/1ec4e99b-ce0d-41d0-baf9-86cabcb6cdee)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/08e68b85-9d25-4c03-82c2-371ef9cde709)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/c64169d3-91e8-4e0c-b11d-7b29b110a66c)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/d49ab057-fcaa-4948-bbcc-279cb37aa880)

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/69f17c22-7bca-4104-96de-a6cfdc6b3e4a)

```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/f524042b-35b0-4083-8fd5-3e0422feae06)

```
dt = pd.read_csv("/content/drive/My Drive/titanic_dataset.csv")
dt

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"] = qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

![download](https://github.com/user-attachments/assets/cf5b9bff-3a93-423b-81f9-07b4fdbdd17f)


# RESULT:
      Thus the given data is performed using Feature Encoding and Transformation process .

       
