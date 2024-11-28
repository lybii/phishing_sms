from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('D:\\phishing_sms\\code\\model_sms.pkl')
tfidf = joblib.load('D:\\phishing_sms\\code\\vectorizer_sms.pkl')

# Initialize Flask app
app = Flask(__name__)

ps = PorterStemmer()

# Function to clean and preprocess SMS
def cleaned_data(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input SMS
    sms = request.form['sms']
    # Preprocess SMS
    cleaned_sms = cleaned_data(sms)
    features = tfidf.transform([cleaned_sms]).toarray()
    # Predict using the model
    prediction = model.predict(features)[0]
    # Return result
    result = 'Lừa đảo' if prediction == 1 else 'Không lừa đảo'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
