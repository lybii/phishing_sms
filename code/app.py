from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    'random_forest':       joblib.load(os.path.join(BASE_DIR, 'model_random_forest.pkl')),
    'decision_tree':       joblib.load(os.path.join(BASE_DIR, 'model_decision_tree.pkl')),
    'naive_bayes':         joblib.load(os.path.join(BASE_DIR, 'model_naive_bayes.pkl')),
    'logistic_regression': joblib.load(os.path.join(BASE_DIR, 'model_logistic_regression.pkl')),
}
MODEL_LABELS = {
    'random_forest':       'Random Forest',
    'decision_tree':       'Decision Tree',
    'naive_bayes':         'Naive Bayes',
    'logistic_regression': 'Logistic Regression',
}
tfidf = joblib.load(os.path.join(BASE_DIR, 'vectorizer_sms.pkl'))

METRICS = {
    'random_forest':      {'accuracy':97.49,'f1':89.78,'precision':99.19,'recall':82.0,  'cm':[[964,1],[27,123]]},
    'decision_tree':      {'accuracy':96.59,'f1':87.42,'precision':86.84,'recall':88.0,  'cm':[[945,20],[18,132]]},
    'naive_bayes':        {'accuracy':96.14,'f1':83.27,'precision':100.0,'recall':71.33, 'cm':[[965,0],[43,107]]},
    'logistic_regression':{'accuracy':96.14,'f1':83.52,'precision':98.2, 'recall':72.67, 'cm':[[963,2],[41,109]]},
}

app = Flask(__name__)
ps  = PorterStemmer()

PHISHING_KEYWORDS = [
    'free','win','winner','won','prize','claim','cash','award',
    'urgent','congratulation','selected','click','verify','account',
    'password','bank','credit','loan','offer','limited','expire',
    'call now','text now','reply','unsubscribe','stop','cancel',
    'guarantee','risk free','bonus','reward','promotion','discount',
    'voucher','coupon','gift','exclusive','special','deal',
]

def cleaned_data(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    try:
        tokens = nltk.word_tokenize(text)
        stop   = stopwords.words('english')
    except Exception:
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        stop   = []
    tokens = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop]
    return ' '.join(tokens)

def highlight_keywords(text):
    return list({kw for kw in PHISHING_KEYWORDS if kw in text.lower()})

def get_confidence(model, features):
    if hasattr(model, 'predict_proba'):
        return round(float(model.predict_proba(features)[0][1]) * 100, 1)
    return None

@app.route('/')
def home():
    return render_template('index.html', model_labels=MODEL_LABELS)

@app.route('/predict', methods=['POST'])
def predict():
    sms       = request.form.get('sms', '').strip()
    model_key = request.form.get('model', 'random_forest')
    if model_key not in MODELS:
        model_key = 'random_forest'
    if not sms:
        return render_template('index.html', error='Vui lòng nhập nội dung SMS.',
                               model_labels=MODEL_LABELS, selected_model=model_key)
    model      = MODELS[model_key]
    features   = tfidf.transform([cleaned_data(sms)]).toarray()
    prediction = model.predict(features)[0]
    is_phishing = bool(prediction == 1)
    return render_template(
        'index.html',
        prediction=    'Lừa đảo' if is_phishing else 'Không lừa đảo',
        is_phishing=   is_phishing,
        confidence=    get_confidence(model, features),
        keywords=      highlight_keywords(sms),
        sms_input=     sms,
        model_labels=  MODEL_LABELS,
        selected_model=model_key,
        model_name=    MODEL_LABELS[model_key],
    )

@app.route('/metrics')
def metrics():
    return render_template('metrics.html', metrics=METRICS, model_labels=MODEL_LABELS)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data      = request.get_json(force=True, silent=True) or {}
    sms       = data.get('sms', '').strip()
    model_key = data.get('model', 'random_forest')
    if model_key not in MODELS:
        model_key = 'random_forest'
    if not sms:
        return jsonify({'error': 'Field "sms" is required.'}), 400
    model      = MODELS[model_key]
    features   = tfidf.transform([cleaned_data(sms)]).toarray()
    prediction = model.predict(features)[0]
    is_phishing = bool(prediction == 1)
    return jsonify({
        'sms':         sms,
        'model':       MODEL_LABELS[model_key],
        'prediction':  'phishing' if is_phishing else 'ham',
        'label_vi':    'Lừa đảo' if is_phishing else 'Không lừa đảo',
        'confidence':  get_confidence(model, features),
        'keywords':    highlight_keywords(sms),
        'is_phishing': is_phishing,
    })

@app.route('/api/metrics')
def api_metrics():
    return jsonify(METRICS)

if __name__ == '__main__':
    app.run(debug=True)