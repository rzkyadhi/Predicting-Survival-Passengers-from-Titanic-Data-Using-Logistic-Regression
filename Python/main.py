import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as pyplot

data_train = pd.read_csv("../Dataset/train.csv", sep=",")
data_test = pd.read_csv("../Dataset/test.csv", sep=",")

data_train.dropna(subset=["Age"], inplace=True)
data_test.dropna(subset=["Age"], inplace=True)

data_train_data = data_train[["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch"]]
data_train_predict = "Survived"

x = np.array(data_train_data.drop([data_train_predict], 1))
y = np.array(data_train_data[data_train_predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Declare Logistic Regression Method
linear_regression = LogisticRegression()

#Declare the Models
linear_regression.fit(x_train, y_train)
accuracy = linear_regression.score(x_test, y_test)
print(accuracy)

predict = linear_regression.predict(x_test)

#Testing
confusionMatrix = pd.DataFrame(confusion_matrix(y_test, predict), columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"])
print(confusionMatrix)

print(classification_report(y_test, predict))