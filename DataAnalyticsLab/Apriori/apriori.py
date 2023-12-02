from itertools import combinations

def load_transactions(filename):
    transactions = []
    with open(filename, 'r') as file:
        for line in file:
            transaction = set(line.strip().split())
            transactions.append(transaction)
    return transactions

def get_unique_items(transactions):
    unique_items = set()
    for transaction in transactions:
        unique_items.update(transaction)
    return unique_items

def get_frequent_itemsets(transactions, min_support):
    unique_items = get_unique_items(transactions)
    itemsets = [{item} for item in unique_items]
    frequent_itemsets = []

    while itemsets:
        next_itemsets = []
        for itemset in itemsets:
            support = sum(1 for transaction in transactions if itemset.issubset(transaction))
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
                next_itemsets.extend([itemset.union({item}) for item in unique_items if item not in itemset])

        itemsets = next_itemsets

    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                antecedent = set(combinations(itemset, i))
                consequent = itemset.difference(antecedent)
                confidence = support / get_support(frequent_itemsets, antecedent)
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules

def get_support(frequent_itemsets, itemset):
    for item, support in frequent_itemsets:
        if item == itemset:
            return support
    return 0

def print_results(frequent_itemsets, rules):
    print("Frequent Itemsets:")
    for itemset, support in frequent_itemsets:
        print(f"{itemset}: {support}")

    print("\nAssociation Rules:")
    for antecedent, consequent, confidence in rules:
        print(f"{antecedent} => {consequent} : {confidence}")

if __name__ == "__main__":
    filename = 'C:\Users\vishn\Desktop\DataAnalyticsLab\DataAnalyticsLab\Apriori\apriori.txt'  # Replace with the path to your transaction data file
    min_support = 2
    min_confidence = 0.7

    transactions = load_transactions(filename)
    frequent_itemsets = get_frequent_itemsets(transactions, min_support)
    rules = generate_association_rules(frequent_itemsets, min_confidence)

    print_results(frequent_itemsets, rules)
