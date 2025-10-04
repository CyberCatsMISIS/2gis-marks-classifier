import pickle

with open("app/marks_classifier.pkl", "rb") as f:
    model_data = pickle.load(f)

vectorizer = model_data["vectorizer"]
clf = model_data["classifier"]
tag_list = model_data["tags"]


def predict_tags(text: str):
    X_new = vectorizer.transform([text])
    y_pred = clf.predict(X_new)[0]
    predicted_tags = [tag_list[i] for i, val in enumerate(y_pred) if val == 1]
    return predicted_tags
