from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Hyperparameters for Random Forest Classifier
# n_estimators: The number of trees in the forest. A common default is 100, but you can experiment with values like 10, 50, or 200.
# max_depth: The maximum depth of the trees. This can help prevent overfitting.
# random_state: A seed for reproducibility. Setting this to a fixed value (e.g., 42) ensures that you get the same results each time you run the code.
# n_jobs: The number of CPU cores to use for training. Setting this to -1 will use all available cores, which can speed up training on larger datasets.
# random_state is set to 42 for reproducibility, but you can choose any integer value or omit it for non-deterministic results. Adjust n_estimators and max_depth based on the size of your dataset and computational resources.


def _to_2d_features(X):
    """Convert image tensors to 2D feature matrix expected by sklearn estimators."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X

def run_validate(rf, X_train, y_train, X_val, y_val):

    X_train = _to_2d_features(X_train)
    X_val = _to_2d_features(X_val)
    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()
    # Train the Random Forest classifier
    print("Training Random Forest classifier...")
    rf.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = rf.predict(X_val)

    # Print classification report and confusion matrix
    rf_acc = rf.score(X_val, y_val)
    print(f"Random Forest accuracy: {rf_acc:.4f}  ({rf_acc*100:.2f} %)")
    print("Random Forest Classifier:")
    print(classification_report(y_val, y_pred))