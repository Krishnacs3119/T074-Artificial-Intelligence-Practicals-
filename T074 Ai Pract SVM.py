import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("enterprise-customers-1000.csv")

df = df[df["Support Level"].isin(["Closed Won", "Closed Lost"])]

X = df[["Number of Employees", "Contract Value"]]
y = df["Support Level"]

X = X.fillna(X.mean())

y = y.map({
    "Closed Lost": 0,
    "Closed Won": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto"],
    "kernel": ["linear", "rbf"]
}

grid_search = GridSearchCV(
    SVC(),
    parameters,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)

model = grid_search.best_estimator_

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("Test Accuracy Percentage:", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

classes = ["Closed Lost", "Closed Won"]

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("SVM Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.colorbar()

plt.xticks(range(len(classes)), classes, rotation=20)
plt.yticks(range(len(classes)), classes)

plt.show()

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1

y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolors="k",
    label="Training Data"
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    marker="x",
    s=80,
    label="Test Data"
)

plt.xlabel("Standardized Number of Employees")
plt.ylabel("Standardized Contract Value")

plt.title("SVM Decision Boundary")

plt.legend()

plt.show()
