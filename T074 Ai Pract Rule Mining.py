import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv(r"C:\Users\PRIYANKA\Downloads\gta_data_batch_1.csv")

columns = [
    "vehicle_class",
    "manufacturer",
    "acquisition",
    "storage_location",
    "delivery_method",
    "modifications",
    "race_availability",
    "bulletproof"
]

transactions = []

for _, row in df.iterrows():
    items = []

    for col in columns:
        if col in df.columns and pd.notna(row[col]):
            items.append(col + "=" + str(row[col]))

    if "features" in df.columns and pd.notna(row["features"]):
        features = str(row["features"]).split(",")
        for feature in features:
            feature = feature.strip()
            if feature:
                items.append("Feature=" + feature)

    transactions.append(items)

te = TransactionEncoder()

transaction_data = te.fit(transactions).transform(transactions)

transaction_df = pd.DataFrame(
    transaction_data,
    columns=te.columns_
)

frequent_itemsets = apriori(
    transaction_df,
    min_support=0.08,
    use_colnames=True
)

frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
)

print("Frequent Itemsets:")
print(frequent_itemsets.head(20))

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

rules = rules.sort_values(
    by=["confidence", "lift"],
    ascending=False
)

print("\nAssociation Rules:")
print(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20)
)

single_itemsets = frequent_itemsets[
    frequent_itemsets["itemsets"].apply(lambda x: len(x) == 1)
].head(10).copy()

single_itemsets["itemsets"] = single_itemsets["itemsets"].apply(
    lambda x: list(x)[0]
)

plt.figure(figsize=(10, 6))

plt.barh(
    single_itemsets["itemsets"],
    single_itemsets["support"]
)

plt.title("Top 10 Frequent Items")
plt.xlabel("Support")
plt.ylabel("Items")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

top_rules = rules.head(10).copy()

top_rules["rule"] = top_rules.apply(
    lambda x: (
        " + ".join(list(x["antecedents"]))
        + " -> "
        + " + ".join(list(x["consequents"]))
    ),
    axis=1
)

top_rules = top_rules.sort_values(
    by="confidence",
    ascending=True
)

plt.figure(figsize=(12, 7))

plt.barh(
    top_rules["rule"],
    top_rules["confidence"]
)

plt.title("Top 10 Association Rules")
plt.xlabel("Confidence")
plt.ylabel("Rules")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
