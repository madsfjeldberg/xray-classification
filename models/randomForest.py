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

def test(rf, X_test, y_test):
    
    X_test = _to_2d_features(X_test)
    y_test = np.asarray(y_test).ravel()

    print("Testing Random Forest classifier...")

    # Predict on the test set
    y_pred = rf.predict(X_test)

    # Print classification report and confusion matrix
    rf_acc = rf.score(X_test, y_test)
    print(f"Random Forest accuracy: {rf_acc:.4f}  ({rf_acc*100:.2f} %)")
    print("Random Forest Classifier:")
    print(classification_report(y_test, y_pred))