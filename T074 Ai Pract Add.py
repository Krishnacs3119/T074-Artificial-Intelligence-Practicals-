import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r"C:\Users\PRIYANKA\Downloads\cs448b_ipasn.csv")

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

X = df.drop(["date", "f"], axis=1)
y = (df["f"] == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weak_classifier = DecisionTreeClassifier(max_depth=1, random_state=42)

weak_classifier.fit(X_train, y_train)

weak_pred = weak_classifier.predict(X_test)

weak_accuracy = accuracy_score(y_test, weak_pred)

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

adaboost.fit(X_train, y_train)

ada_pred = adaboost.predict(X_test)

ada_accuracy = accuracy_score(y_test, ada_pred)

print("Weak Classifier Accuracy:", weak_accuracy)
print("AdaBoost Accuracy:", ada_accuracy)

print("\nWeak Classifier Report:")
print(classification_report(y_test, weak_pred))

print("\nAdaBoost Report:")
print(classification_report(y_test, ada_pred))

models = ["Weak Classifier", "AdaBoost"]
accuracies = [weak_accuracy, ada_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)
plt.title("Weak Classifier vs AdaBoost")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.02, f"{value:.4f}", ha="center")

plt.show()
