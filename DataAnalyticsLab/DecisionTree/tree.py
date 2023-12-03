import pandas as pd

def fit(X, y, max_depth=None, depth=0):
    if len(set(y)) == 1 or (max_depth is not None and depth == max_depth):
        return {'class': y.iloc[0]}

    best_feature, best_threshold = find_best_split(X, y)

    if best_feature is None:
        return {'class': y.mode().iloc[0]}

    left_indices = X[best_feature] <= best_threshold
    right_indices = ~left_indices

    left_subtree = fit(X[left_indices], y[left_indices], max_depth, depth + 1)
    right_subtree = fit(X[right_indices], y[right_indices], max_depth, depth + 1)

    return {'feature_index': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree}

def find_best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature in X.columns:
        thresholds = set(X[feature])
        for threshold in thresholds:
            left_indices = X[feature] <= threshold
            right_indices = ~left_indices

            gini = calculate_gini_index(y[left_indices], y[right_indices])
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

def calculate_gini_index(left_labels, right_labels):
    left_size = len(left_labels)
    right_size = len(right_labels)
    total_size = left_size + right_size

    if total_size == 0:
        return 0

    p_left = left_size / total_size
    p_right = right_size / total_size

    gini_left = 1 - (left_labels.value_counts(normalize=True) ** 2).sum()
    gini_right = 1 - (right_labels.value_counts(normalize=True) ** 2).sum()

    gini_index = p_left * gini_left + p_right * gini_right
    return gini_index

def predict(X, tree):
    if 'class' in tree:
        return tree['class']

    feature_index = tree['feature_index']
    threshold = tree['threshold']

    if X[feature_index] <= threshold:
        return predict(X, tree['left'])
    else:
        return predict(X, tree['right'])


# Load data from CSV file using pandas
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Labels
    return X, y


# Example usage
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with the path to your CSV file
    dataset_path = 'DataAnalyticsLab\DecisionTree\Tree.csv'
    X_train, y_train = load_data(dataset_path)

    # Create and train the decision tree
    max_depth = 3
    tree_model = fit(X_train, y_train, max_depth)

    # Example prediction
    example_instance = X_train.iloc[0]
    prediction = predict(example_instance, tree_model)
    print(f"Prediction for {example_instance}: {prediction}")
