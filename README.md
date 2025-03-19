# Ex.No-05-Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
 
## Program:
Name : Piritharaman R

Reg no : 212223230148

```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
## Output:
![image](https://github.com/user-attachments/assets/3e4e61d7-07fc-4010-b67e-4a528d517ef4)

```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()
```
![image](https://github.com/user-attachments/assets/9e0f98c2-7980-410d-87c0-8810fdc1f1bf)
```
data1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/3416e320-ae11-4199-88bc-cbe87d488cba)
```
data1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/a40289fc-c1ff-4dcd-8e85-851154030d45)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 
```
![image](https://github.com/user-attachments/assets/742b92d9-e8e6-4519-9f53-62af971ca1a8)
```
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/cfb9dfd6-3c61-4fb8-9a33-a2864da1de43)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/535e4b37-406c-47a6-922f-3b1632355024)
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/955c1a80-72f2-4b5a-9922-69ef3ca128db)

```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/6f11334a-e0ad-4be7-b175-5ad83ffa8e9f)

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/a0fe45ba-ef4f-4e1f-ac2e-6695a686a62e)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
