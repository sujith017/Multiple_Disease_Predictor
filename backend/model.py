import numpy as np
import pandas as pd
import joblib  # For saving models and scalers
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("backend\data_set.csv")

y = df.iloc[:, -1].values  
X = df.iloc[:, :-1].values


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

rf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', probability=True, random_state=0)


ensemble_model = VotingClassifier(
    estimators=[('RandomForest', rf), ('NaiveBayes', nb), ('KNN', knn), ('SVM', svm)],
    voting='soft'
)


ensemble_model.fit(X_train, y_train)

joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

y_pred = ensemble_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
