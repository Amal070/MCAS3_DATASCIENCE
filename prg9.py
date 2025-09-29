import numpy as np
import pandas as pd
import seaborn as  sns

iris = sns.load_dataset("iris")
df = pd.DataFrame(iris)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("\nPredicted Labels", y_pred,"\n")
print("\nActual Labels", y_test,"\n")
from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("\n Accuracy", accuracy_score(y_test,y_pred))

mislabeled_count = (y_test != y_pred).sum()
print(f"\n Number of mislabeled points: {mislabeled_count} out of {len(y_test)}")
mismatches = (y_test != y_pred)
for actual,predicted in zip(y_test[mismatches],y_pred[mismatches]):
    # print(f"\n Actual labels: {actual}\n")
    print(f"Predicted labels: {predicted}\n")


