import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the preprocessed hand sign dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Reshape the data to 2D
data_shape = data.shape
data = data.reshape(data_shape[0], -1)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate the trained model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print('Random Forest Training Summary:')
print('--------------------------------')
print('Number of training samples:', x_train.shape[0])
print('Number of testing samples:', x_test.shape[0])
print('Number of features:', x_train.shape[1])
print('Number of classes:', len(np.unique(labels)))
print('--------------------------------')
print('Trained Accuracy: {:.2f}%'.format(accuracy * 100))
print('--------------------------------')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print('Trained Random Forest was successfully saved to "model.p"!')
