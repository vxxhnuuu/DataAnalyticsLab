import numpy as np
import pandas as pd

def calculate_gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini_impurity = 1 - np.sum(probabilities ** 2)
    return gini_impurity

def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def find_best_split(X, y):
    m, n = X.shape
    initial_gini = calculate_gini_impurity(y)
    best_gini = float('inf')
    best_feature_index = None
    best_threshold = None

    for feature_index in range(n):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gini_left = len(y_left) / m * calculate_gini_impurity(y_left)
            gini_right = len(y_right) / m * calculate_gini_impurity(y_right)
            weighted_gini = gini_left + gini_right
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

def build_tree(X, y, depth):
    unique_classes, counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]

    if depth == 0 or len(unique_classes) == 1:
        return {'value': majority_class}

    feature_index, threshold = find_best_split(X, y)

    if feature_index is None:
        return {'value': majority_class}

    X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)

    left_subtree = build_tree(X_left, y_left, depth - 1)
    right_subtree = build_tree(X_right, y_right, depth - 1)

    return {'feature_index': feature_index, 'threshold': threshold,
            'left': left_subtree, 'right': right_subtree}

def fit(X, y, max_depth=None):
    return build_tree(X, y, max_depth)

def predict_instance(node, x):
    if 'value' in node:
        return node['value']
    if x[node['feature_index']] <= node['threshold']:
        return predict_instance(node['left'], x)
    else:
        return predict_instance(node['right'], x)

def predict(tree, X):
    return [predict_instance(tree, x) for x in X]

# Load data
data_df = pd.read_csv('DataAnalyticsLab\DecisionTree\Tree.csv')
print("Dataset:\n", data_df)

X = data_df.iloc[:, :-1].values
y = data_df.iloc[:, -1].values

# Train the decision tree
dt_tree = fit(X, y, max_depth=3)

# Make predictions
predictions = predict(dt_tree, X)
print("Predictions:", predictions)
