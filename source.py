import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm

# Đọc dữ liệu từ file csv chứa email và nhãn
data = pd.read_csv('spam.csv')
emails = data['text']
labels = data['label']
email_train, email_test,lebels_train, lebels_test = train_test_split(emails,labels,test_size = 0.2)

# Tạo vector đặc trưng từ các email
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(emails)

# Tạo mô hình bộ lọc Bayes Naive
nb = MultinomialNB()
nb.fit(features, labels)

# Sử dụng mô hình để dự đoán nhãn của một email mới
new_email = ["Bạn đã thắng giải thưởng lớn"]
new_features = vectorizer.transform(new_email)
prediction = nb.predict(new_features)
print(prediction)
