import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier

# Заглушка: пример обученной модели
vectorizer = CountVectorizer()
clf = MultiOutputClassifier(RandomForestClassifier())


# Функция предсказания тегов
def predict_tags(text: str):
    X = vectorizer.transform([text])
    # Заменить на реальные метки
    y_pred = clf.predict(X)
    # Преобразовать в список строк
    return ["tag1", "tag2"]  # заглушка
