import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load your transaction data into a DataFrame
data = pd.read_csv('transaction_data.csv')

# Perform one-hot encoding to create a binary transaction-item matrix
basket = pd.get_dummies(data, columns=['item'])

# Use Apriori to find frequent item sets
frequent_item_sets = apriori(basket, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_item_sets, metric='lift', min_threshold=1.0)

# Filter and prioritize rules based on support, confidence, and lift
filtered_rules = rules[(rules['support'] > 0.2) & (rules['confidence'] > 0.7) & (rules['lift'] > 1.2)]

# Print the top rules
print(filtered_rules.head())
