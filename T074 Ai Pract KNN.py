import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(r"C:\Users\PRIYANKA\Downloads\college_sleep_and_gpa.csv")

df = df.drop(
    ["student_id", "sleep_midpoint_clock", "avg_sleep_minutes",
     "avg_sleep_hours", "under_6h_sleep"],
    axis=1
)

X = df.drop("sleep_bracket", axis=1)
y = df["sleep_bracket"]

for column in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].fillna("Unknown"))

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

le_y = LabelEncoder()
y = le_y.fit_transform(y)

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

k_values = list(range(1, 21))
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, prediction))

best_k = k_values[accuracies.index(max(accuracies))]

print("Best K Value:", best_k)
print("Best Accuracy:", max(accuracies))

knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nK-NN Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=le_y.classes_,
        zero_division=0
    )
)

print("\nPredicted Classes:")
print(le_y.inverse_transform(y_pred[:10]))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le_y.classes_
)

disp.plot()
plt.title("K-NN Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o")
plt.title("K-NN Accuracy for Different K Values")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid()
plt.tight_layout()
plt.show()
