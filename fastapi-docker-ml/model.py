import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from dataLoader import *
import joblib

def printResult(y_pred, y_prob):
    acc = accuracy_score(test_data["label"], y_pred)
    # Result
    print("Accuracy: {:.2f}".format(acc*100),end='\n\n')
    cm = confusion_matrix(test_data["label"],y_pred)
    print('Confusion Matrix:\n', cm)
    print(classification_report(test_data["label"],y_pred))

if __name__ == '__main__':
    model = MLPClassifier(solver='adam', alpha=2e-4, hidden_layer_sizes=(5, 2), max_iter=700)

    model.fit(bow_train_features, y_resampled)

    joblib.dump(model,'model.joblib')
    joblib.dump(vectorizer,'vectorizer.joblib')
    y_pred_bow_mlp = model.predict(bow_test_features.toarray())

    y_prob_bow_mlp = model.predict_proba(bow_test_features.toarray())[:,1]

    printResult(y_pred_bow_mlp, y_prob_bow_mlp)



