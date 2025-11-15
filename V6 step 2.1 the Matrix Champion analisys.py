# ==============================================================================
# Step 2.3: Evaluating Model Performance
# ==============================================================================
# This script performs the following actions:
# 1. Loads the best saved model and the unseen test (validation) data.
# 2. Makes predictions on the test data.
# 3. Generates a detailed Classification Report (Precision, Recall, F1-Score).
# 4. Creates and displays a visual Confusion Matrix to identify where the model
#    is making mistakes.
# ==============================================================================

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- 1. Load the Model and Test Data ---

print("--- Loading the trained model and test data ---")

try:
    # Load the saved Keras model
    model = tf.keras.models.load_model('final_psychology_model.keras')
    # Load the test data arrays
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    # Load the label names for our reports
    label_classes = np.load('label_encoder.npy', allow_pickle=True)
except (FileNotFoundError, IOError) as e:
    print(f"FATAL ERROR: Could not load necessary files. {e}")
    print("Please ensure 'best_psychology_model.keras' and the .npy files are in the directory.")
    sys.exit(1)

print("Model and data loaded successfully.")
print(f"Test data shape: {X_test.shape}")

# --- 2. Make Predictions on the Test Data ---

print("\n--- Making predictions on the unseen test data ---")

# The model outputs probabilities for each class (e.g., [0.1, 0.05, 0.7, 0.1, 0.05])
y_pred_probs = model.predict(X_test)

# We take the index of the highest probability as our final prediction
# For [0.1, 0.05, 0.7, 0.1, 0.05], the index is 2, which corresponds to 'Herd'
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# --- 3. Generate the Classification Report ---

print("\n--- Classification Report ---")
# This report provides a detailed breakdown of the model's performance for each class
# - Precision: Of all the times the model predicted a class, how often was it correct?
# - Recall: Of all the actual instances of a class, how many did the model correctly identify?
# - F1-Score: A weighted average of Precision and Recall.
report = classification_report(y_test, y_pred_classes, target_names=label_classes)
print(report)

# --- 4. Generate and Visualize the Confusion Matrix ---

print("\n--- Generating Confusion Matrix ---")
# The confusion matrix shows how many times the model confused one class for another.
# The rows are the TRUE labels, and the columns are the PREDICTED labels.
cm = confusion_matrix(y_test, y_pred_classes)

# Plotting the confusion matrix for better visual interpretation
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_classes, yticklabels=label_classes)

plt.title('Confusion Matrix: Model Predictions vs. True Labels', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()

print("\n--- Step 2.3 Complete ---")
print("Review the Classification Report and the Confusion Matrix plot to assess the model's performance.")