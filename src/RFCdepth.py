from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier
from src.model import X_train, X_test, y_train, y_test


def RF_accuracy(n):
    rfc = RandomForestClassifier(criterion="gini",random_state = 42, max_depth=n)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    acc=accuracy_score(y_test, predictions)
    return acc