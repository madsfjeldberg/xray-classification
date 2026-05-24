from sklearn.ensemble import RandomForestClassifier
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

def run_validate(rf, X_train, y_train, X_val, y_val):

    X_train = _to_2d_features(X_train)
    X_val = _to_2d_features(X_val)
    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()
    _validate_dataset(X_train, y_train, "train")
    _validate_dataset(X_val, y_val, "val")
    # Train the Random Forest classifier
    print("Training Random Forest classifier...")
    rf.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = rf.predict(X_val)

    # Print classification report and confusion matrix
    rf_acc = rf.score(X_val, y_val)
    print(f"Random Forest accuracy: {rf_acc:.4f}  ({rf_acc*100:.2f} %)")
    print("Random Forest Classifier:")
    print(classification_report(y_val, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

def test(rf, X_test, y_test):
    
    X_test = _to_2d_features(X_test)
    y_test = np.asarray(y_test).ravel()
    _validate_dataset(X_test, y_test, "test")

    print("Testing Random Forest classifier...")

    # Predict on the test set
    y_pred = rf.predict(X_test)

    # Print classification report and confusion matrix
    rf_acc = rf.score(X_test, y_test)
    print(f"Random Forest accuracy: {rf_acc:.4f}  ({rf_acc*100:.2f} %)")
    print("Random Forest Classifier:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"]))