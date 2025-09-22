import numpy as np
import pandas as pd

dataset = pd.read_csv("iris (2).csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(y_test, y_pred))
5
df =pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df)

new_test_point = np.array([[5.1, 3.1, 1.4, 0.2]])
prediction = classifier.predict(new_test_point)
print(f"\n Prediction : {prediction[0]}")