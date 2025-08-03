import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("backend/psl_sentence_dataset.csv")
df = df.dropna()

X = df.iloc[:, :-1].astype(float)
y = df.iloc[:, -1].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("✅ Sentence model accuracy:", model.score(X_test, y_test))

joblib.dump(model, "backend/psl_sentence_model.pkl")
print("✅ Saved model: backend/psl_sentence_model.pkl")
