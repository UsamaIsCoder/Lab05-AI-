# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Load the Pakistani cricket players data using NumPy
data = np.array([
    ["Babar Azam", "male", 5.11, 165, 10],
    ["Shaheen Shah Afridi", "male", 6.04, 180, 11],
    ["Mohammad Rizwan", "male", 5.74, 160, 9],
    ["Fakhar Zaman", "male", 5.11, 170, 10],
    ["Sarfaraz Ahmed", "male", 5.07, 145, 8],
    ["Nida Dar", "female", 5.04, 120, 6],
    ["Javeria Khan", "female", 5.06, 125, 7],
    ["Bismah Maroof", "female", 5.05, 130, 7],
])

# Step 2: Encode the target column (Gender)
# Mapping: male -> 1, female -> 0
gender_map = {"male": 1, "female": 0}
y = np.array([gender_map[row[1]] for row in data])

# Extract features (Height, Weight, Foot_Size)
X = np.array([[float(row[2]), int(row[3]), int(row[4])] for row in data])

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print Confusion Matrix and Accuracy Score
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy Score: {accuracy:.2f}")

# Step 6: Predict the gender for a new entry
print("\nEnter the details for prediction:")
height = float(input())  # Do not mention "Height" in the prompt
weight = int(input())    # Do not mention "Weight" in the prompt
foot_size = int(input()) # Do not mention "Foot Size" in the prompt

new_entry = np.array([[height, weight, foot_size]])

# Make the prediction
prediction = model.predict(new_entry)
predicted_gender = "male" if prediction[0] == 1 else "female"

# Print only the predicted gender
print(f"The predicted gender for the new entry is: {predicted_gender}")
