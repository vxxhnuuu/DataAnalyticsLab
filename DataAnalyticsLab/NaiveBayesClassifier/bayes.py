def train_naive_bayes_classifier(training_data):
    class_probs = {}
    word_probs = {}

    total_documents = len(training_data)

    # Calculate class probabilities
    for document, label in training_data:
        if label not in class_probs:
            class_probs[label] = 1
        else:
            class_probs[label] += 1

    for label in class_probs:
        class_probs[label] /= total_documents

    # Calculate word probabilities
    word_counts = {}
    for document, label in training_data:
        for word in document.split():
            if (word, label) not in word_counts:
                word_counts[(word, label)] = 1
            else:
                word_counts[(word, label)] += 1

    for (word, label), count in word_counts.items():
        if label not in word_probs:
            word_probs[label] = {}

        word_probs[label][word] = count

    return class_probs, word_probs

def classify_document(document, class_probs, word_probs):
    best_label = None
    max_prob = float('-inf')

    for label in class_probs:
        prob = class_probs[label]

        for word in document.split():
            if label in word_probs and word in word_probs[label]:
                prob *= word_probs[label][word]
            else:
                prob *= 1e-10  # small constant to avoid zero probability

        if prob > max_prob:
            max_prob = prob
            best_label = label

    return best_label

# Sample training data
training_data = [
    ('I love this product', 'positive'),
    ('This is an amazing product', 'positive'),
    ('I hate this product', 'negative'),
    ('This product is terrible', 'negative'),
    ('I have no opinion about this product', 'neutral'),
    ('It is okay, not great, not terrible', 'neutral'),
]

# Sample test data
test_data = [
    ('This is a fantastic product', 'positive'),
    ('I dislike this item', 'negative'),
    ('I have mixed feelings about this product', 'neutral'),
]

# Train the classifier
class_probs, word_probs = train_naive_bayes_classifier(training_data)

# Test the classifier
correct_predictions = 0
for doc, true_label in test_data:
    predicted_label = classify_document(doc, class_probs, word_probs)
    print(f"Actual: {true_label}, Predicted: {predicted_label}")
    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)
print(f"Accuracy: {accuracy}")
