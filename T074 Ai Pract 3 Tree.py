import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('GamingStudy_data.csv', encoding='latin-1')

features = ['Hours', 'Age', 'SWL_T', 'SPIN_T', 'Narcissism']
target = 'GAD_T'

df_clean = df.dropna(subset=features + [target]).copy()

X = df_clean[features]
y = (df_clean[target] > 9).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Anxiety (<=9)', 'High Anxiety (>9)']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(16, 10))
plot_tree(
    clf, 
    feature_names=features, 
    class_names=['Low Anxiety', 'High Anxiety'], 
    filled=True, 
    rounded=True, 
    fontsize=11,
    proportion=True
)
plt.title("Decision Tree: Predicting High Anxiety in Gamers", fontsize=16)
plt.tight_layout()
plt.show()
