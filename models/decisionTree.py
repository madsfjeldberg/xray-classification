from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def _to_2d_features(X):
    """Convert image tensors to 2D feature matrix expected by sklearn estimators."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X


def _validate_dataset(X, y, split_name):
    if X.shape[0] == 0:
        raise ValueError(
            f"'{split_name}' dataset is empty after preprocessing. "
            "Verify split name and image files."
        )
    if y.shape[0] == 0:
        raise ValueError(
            f"'{split_name}' labels are empty after preprocessing. "
            "Verify split name and label loading."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sample/label mismatch for '{split_name}': {X.shape[0]} samples vs {y.shape[0]} labels."
        )

def run_validate(dt, X_train, y_train, X_eval, y_eval):

    X_train = _to_2d_features(X_train)
    X_eval = _to_2d_features(X_eval)
    y_train = np.asarray(y_train).ravel()
    y_eval = np.asarray(y_eval).ravel()
    _validate_dataset(X_train, y_train, "train")
    _validate_dataset(X_eval, y_eval, "eval")
    # Train the Decision Tree classifier
    print("Training Decision Tree classifier...")
    dt.fit(X_train, y_train)

    # Predict on the evaluation set
    y_pred = dt.predict(X_eval)

    # Print classification report and confusion matrix
    dt_acc = dt.score(X_eval, y_eval)
    print(f"Decision Tree accuracy: {dt_acc:.4f}  ({dt_acc*100:.2f} %)")
    print("Decision Tree Classifier:")
    print(classification_report(y_eval, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

def test(dt, X_test, y_test):
    
    X_test = _to_2d_features(X_test)
    y_test = np.asarray(y_test).ravel()
    _validate_dataset(X_test, y_test, "test")

    # Test a Decision Tree classifier
    print("Testing Decision Tree classifier...")

    # Predict on the test set
    y_pred = dt.predict(X_test)

    # Print classification report and confusion matrix
    dt_acc = dt.score(X_test, y_test)
    print(f"Decision Tree accuracy: {dt_acc:.4f}  ({dt_acc*100:.2f} %)")
    print("Decision Tree Classifier:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"]))