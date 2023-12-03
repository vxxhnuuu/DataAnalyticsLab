def train_naive_bayes_classifier(data):
    # Separate data by class
    class_data = {}
    for entry in data:
        features, label = entry[:-1], entry[-1]
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(features)

    # Calculate class probabilities and feature probabilities
    class_probabilities = {}
    feature_probabilities = {}

    total_entries = len(data)
    for label, features_list in class_data.items():
        class_probabilities[label] = len(features_list) / total_entries

        feature_probabilities[label] = {}
        for feature_index in range(len(features_list[0])):
            feature_values = [entry[feature_index] for entry in features_list]
            unique_values = set(feature_values)
            feature_probabilities[label][feature_index] = {
                value: feature_values.count(value) / len(features_list)
                for value in unique_values
            }

    return class_probabilities, feature_probabilities


def predict_naive_bayes(class_probabilities, feature_probabilities, input_features):
    best_label = None
    best_score = -1

    for label, class_probability in class_probabilities.items():
        score = class_probability

        for feature_index, feature_value in enumerate(input_features):
            if feature_value in feature_probabilities[label][feature_index]:
                score *= feature_probabilities[label][feature_index][feature_value]
            else:
                # Laplace smoothing for unseen values
                score *= 1e-5

        if score > best_score:
            best_score = score
            best_label = label

    return best_label


# Specify training data
training_data = [
    [1, 35000, 'Yes', 'No'],
    [2, 50000, 'No', 'Yes'],
    [3, 25000, 'Yes', 'Yes'],
    [4, 60000, 'No', 'Yes'],
    [5, 80000, 'Yes', 'Yes'],
    [6, 45000, 'No', 'No'],
    [7, 55000, 'Yes', 'Yes'],
    [8, 20000, 'Yes', 'No'],
    [9, 30000, 'No', 'Yes'],
]

# Train the Naive Bayes classifier
class_probabilities, feature_probabilities = train_naive_bayes_classifier(training_data)

# Get user input for test data
num_test_entries = int(input("Enter the number of test entries: "))
test_data = []
for _ in range(num_test_entries):
    test_entry = input("Enter test entry (space-separated features): ").split()
    test_data.append(test_entry)

# Make predictions on the test data
for test_entry in test_data:
    predicted_label = predict_naive_bayes(class_probabilities, feature_probabilities, test_entry)
    print(f"Predicted label for {test_entry}: {predicted_label}")
