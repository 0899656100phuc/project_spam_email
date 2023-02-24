import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



# Đọc dữ liệu từ file csv chứa email và nhãn
data = pd.read_csv('spam.csv')
emails = data['text']
labels = data['label']

# Tạo vector đặc trưng từ các email
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(emails)

# Tạo mô hình bộ lọc Bayes Naive
nb = MultinomialNB()
nb.fit(features, labels)

# Sử dụng mô hình để dự đoán nhãn của một email mới
new_email = ["bạn đã đăng ký thành công"]
new_features = vectorizer.transform(new_email)
prediction = nb.predict(new_features)
print(prediction)
