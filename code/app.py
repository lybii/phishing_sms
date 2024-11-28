from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Tải xuống các tài nguyên cần thiết
nltk.download('punkt')
nltk.download('stopwords')

# Tải model và vectorizer
model = joblib.load('D:\\phishing_sms\\code\\model_sms.pkl')  # Đường dẫn tới model đã lưu
tfidf = joblib.load('D:\\phishing_sms\\code\\vectorizer_sms.pkl')  # Đường dẫn tới vectorizer

# Khởi tạo Flask app
app = Flask(__name__)

ps = PorterStemmer()

# Hàm để làm sạch và tiền xử lý SMS
def cleaned_data(text):
    text = text.lower()  # Chuyển tất cả sang chữ thường
    text = re.sub('[^a-zA-Z0-9]', ' ', text)  # Loại bỏ ký tự đặc biệt
    text = re.sub(r'http\S+', '', text)  # Loại bỏ đường dẫn URL
    text = nltk.word_tokenize(text)  # Tách từ
    # Loại bỏ stopwords và stemming
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)  # Ghép lại thành chuỗi

# Route trang chủ
@app.route('/')
def home():
    return render_template('index.html')  # Render file HTML giao diện

# Route xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy nội dung SMS người dùng nhập từ form
    sms = request.form['sms']
    # Tiền xử lý nội dung SMS
    cleaned_sms = cleaned_data(sms)
    features = tfidf.transform([cleaned_sms]).toarray()  # Biến đổi nội dung thành vector
    # Dự đoán bằng mô hình
    prediction = model.predict(features)[0]
    # Chuyển kết quả dự đoán thành chuỗi hiển thị
    result = 'Lừa đảo' if prediction == 1 else 'Không lừa đảo'
    # Trả về giao diện với kết quả và nội dung người dùng nhập
    return render_template('index.html', prediction=result, sms_input=sms)

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
