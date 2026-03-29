import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_models(df):

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        score = f1_score(y_test, pred)

        if score > best_score:
            best_score = score
            best_model = model

    return best_model, X