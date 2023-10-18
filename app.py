from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 데이터 로드 및 분석
df = pd.read_csv('movie_reviews.csv')
reviews = df['review']
sentiments = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2)
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_transformed,y_train)

# Flask 웹 애플리케이션 생성
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        review=request.form['review']
        data=[review]
        vect=vectorizer.transform(data).toarray()
        my_prediction=model.predict(vect)
    return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
