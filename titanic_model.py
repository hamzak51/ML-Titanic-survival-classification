# libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Loading the Dataset
df = pd.read_csv("train.csv")


# Analyzing the data
# print(df.info())    #Displaying data types and non-null counts
# print(df.isnull().sum())    #Displaying how many missing values each column has


# Cleaning Data
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
# print(df.columns)   #Checking which columns are left


# Handling missing data
df['Age'].fillna(df['Age'].median(), inplace=True)  #Missing age filled with median value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) #Missing 'Embarked' filled with Mode
# print(df.isnull().sum())  


#Machine Learning
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])           # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked']) # will map S/C/Q to numbers


#Defining variables prediction is based on
X = df[['Pclass', 'Sex', 'Age', 'Fare']]    #independent
y = df['Survived']  #dependent


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print("Training size:", X_train.shape)
# print("Test size:", X_test.shape)


#Scaling
#Ensures model doesn't get biased just because one feature has bigger numbers than another
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Training logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


#Prediction and model evaluation
y_pred = model.predict(X_test)
print("Accuracy %:", accuracy_score(y_test, y_pred)*100, "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



#Graphs
# Graph 1: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# Graph 2: Survival Count by Gender
df_plot = df.copy()
df_plot['Sex'] = df_plot['Sex'].map({0: 'female', 1: 'male'})
sns.countplot(x='Survived', hue='Sex', data=df_plot)
plt.title("Survival Count by Gender")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title='Sex')
plt.show()


# Graph 3: Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title='Survived')
plt.show()


# Graph 4: Age Distribution by Survival
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()




# Graph 5: Survival Percentage by Age Group (Bar Chart)

# Create age bins (you can adjust these)
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
labels = ['0–10', '11–20', '21–30', '31–40', '41–50', '51–60', '61–70', '71–80']

# Create a new column 'AgeGroup'
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Group by AgeGroup and calculate survival rate
age_group_survival = df.groupby('AgeGroup')['Survived'].mean() * 100

# Plot
age_group_survival.plot(kind='bar', color='black')
plt.title("Survival Percentage by Age Group")
plt.ylabel("Survival Rate (%)")
plt.xlabel("Age Group")
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()