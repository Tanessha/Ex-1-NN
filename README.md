<H3>ENTER YOUR NAME: Tanessha Kannan</H3>
<H3>ENTER YOUR REGISTER NO. 212223040225</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```


## OUTPUT:
### Dataset:
![image](https://github.com/user-attachments/assets/87a093de-58e1-4ce2-890f-900bfbf50ddd)

### X Values:
![image](https://github.com/user-attachments/assets/ff0430dc-58a6-4221-bdd9-5d61f43a1659)

### Y Values:
![image](https://github.com/user-attachments/assets/ad643569-f6a2-4c35-9b01-704d7afed2d3)

### Null Values:
![image](https://github.com/user-attachments/assets/8a2ccc3a-448e-45fe-bc33-fc85ef58a47f)

### Duplicated Values:
![image](https://github.com/user-attachments/assets/9084e331-716e-4a7b-81eb-59db71991bf1)

### Description:
![image](https://github.com/user-attachments/assets/fbc82145-b1a1-47f1-af2b-1ecad239168e)

### Normalized Dataset:
![image](https://github.com/user-attachments/assets/aa48d922-0b62-49a6-b4fb-37018f8ae5f9)

### Training Data:
![image](https://github.com/user-attachments/assets/3e2501d8-3b3d-4614-b94c-93d5c6c43a58)

### Testing Data:
![image](https://github.com/user-attachments/assets/50cd3893-ef9f-4acc-bb43-32772452c0e4)
## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


