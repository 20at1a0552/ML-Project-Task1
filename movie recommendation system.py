import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
# DATA HANDLING
dataset = pd.read_csv('Dataset.csv')
movie_title = pd.read_csv('Movie_Id_Titles.csv')
data = pd.merge(dataset, movie_title, on='item_id')

# Use 'rating' as the target variable
X = data[['item_id']]
Y = (data['rating'] >= 3.5).astype(int)  # Binary classification: 1 if rating >= 3.5, else 0

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# LOGISTIC REGRESSION
logreg_model = LogisticRegression()
logreg_model.fit(X_train, Y_train)

# Make predictions on the test set
pred = logreg_model.predict(X_test)

# Evaluate the model using accuracy
accuracy = accuracy_score(Y_test, pred)
conf_matrix = confusion_matrix(Y_test, pred)

print("Merge data is:",data)
print("Accuracy (Logistic Regression): {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", conf_matrix)

# Plot the results (this will be a simple scatter plot for binary classification)
plt.scatter(X_test, Y_test, color='b', label='Actual')
plt.scatter(X_test, pred, color='red', label='Logistic Regression Predictions')
plt.xlabel('item_id')
plt.ylabel('Rating (1: Liked, 0: Disliked)')
plt.legend()
plt.show()