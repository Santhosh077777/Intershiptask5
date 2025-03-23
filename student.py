import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import nltk
nltk.download('punkt_tab')

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- Task 1: Student Pass/Fail Prediction ---
student_df = pd.read_csv('/content/student_data (5).csv')
student_df.dropna(inplace=True)

X = student_df[['Study Hours', 'Attendance']]
y = student_df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Student Performance Prediction Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# --- Task 2: Sentiment Analysis ---
reviews_df = pd.read_csv('/content/reviews.csv')
reviews_df.dropna(inplace=True)

# Apply text preprocessing
reviews_df['Cleaned_Review'] = reviews_df['Review Text'].apply(preprocess_text)

# Convert text data to numerical format
vectorizer = TfidfVectorizer()
X_reviews = vectorizer.fit_transform(reviews_df['Cleaned_Review'])
y_reviews = reviews_df['Sentiment'].map({'positive': 1, 'negative': 0})

# Split data for training
X_train_reviews, X_test_reviews, y_train_reviews, y_test_reviews = train_test_split(X_reviews, y_reviews, test_size=0.2, random_state=42)

# Train a Logistic Regression model
review_model = LogisticRegression()
review_model.fit(X_train_reviews, y_train_reviews)

# Predict sentiments
y_pred_reviews = review_model.predict(X_test_reviews)

# Evaluate the model
accuracy_reviews = accuracy_score(y_test_reviews, y_pred_reviews)
conf_matrix_reviews = confusion_matrix(y_test_reviews, y_pred_reviews)

print("Sentiment Analysis Accuracy:", accuracy_reviews)
print("Confusion Matrix:\n", conf_matrix_reviews)
