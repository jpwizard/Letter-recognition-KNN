#James Panagis


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# loading the the dataset
file_path = r'C:\Users\junko\Downloads\letter+recognition/letter-recognition.data'

column_names = ['letter', 'x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
data = pd.read_csv(file_path, names=column_names)

# assyming that letter is the target variable
X = data.drop('letter', axis=1)
y = data['letter']

# splitting the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# function to calculate the distances
def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2)**2))

def calculate_distances(test_instance, training_data):
    distances = []
    for index, row in training_data.iterrows():
        distance = euclidean_distance(test_instance, row[:-1])  # assuming that the last column is the target variable
        distances.append((index, distance))
    return distances

# funcion to make the predictions based on k neighbors
def predict(test_instance, training_data, training_labels, k):
    distances = calculate_distances(test_instance, training_data)
    distances.sort(key=lambda x: x[1])  # sorting distances in ascending order
    k_nearest_neighbors = distances[:k]  # selecting the top k neighbors

    # extracting the labels of the k neighbors using index values
    k_nearest_labels = [training_labels.loc[index] for index, _ in k_nearest_neighbors]

    # making a prediction based on majority class
    prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
    return prediction

#functio to check the performance of the model using different k values
def evaluate_model(k_values, X_train, X_test, y_train, y_test):
    for k in k_values:
        predictions = []
        for _, test_instance in X_test.iterrows():
            prediction = predict(test_instance, X_train, y_train, k)
            predictions.append(prediction)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        print(f"Metrics for K={k}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\n")

#exxample usage
k_values = [1, 3, 5]  # I know that you can adjust the list based on the values of k you want to evaluate
evaluate_model(k_values, X_train, X_test, y_train, y_test)
