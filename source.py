import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

spam = pd.read_csv('spam.csv', encoding='latin-1')

data = spam.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v2": "text", "v1": "label"})

z = data['text']
y = data["label"]

z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train)

model = svm.SVC()
model.fit(features, y_train)

newEmail = [""]

features_test = cv.transform(z_test)
print(model.score(features_test, y_test))

new_features = cv.transform(newEmail)
prediction = model.predict(new_features)
print(prediction)
