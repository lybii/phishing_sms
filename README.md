# 🛡️ SMS Shield — Phishing SMS Detector

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Ứng dụng web phát hiện tin nhắn SMS lừa đảo (phishing) sử dụng **Machine Learning** và **NLP**.

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 🤖 4 ML Models | Random Forest, Decision Tree, Naive Bayes, Logistic Regression |
| 📊 Model Metrics | Trang `/metrics` hiển thị Accuracy, F1, Precision, Recall, Confusion Matrix |
| 📈 Confidence Score | Xác suất phishing theo % |
| 🏷️ Keyword Detection | Phát hiện 30+ từ khóa nguy hiểm trong SMS |
| ⚡ REST API | Endpoint `POST /api/predict` trả về JSON |
| 🎨 Dark Mode UI | Giao diện hiện đại, responsive |

---

## 📊 Kết quả đánh giá (Test set 20% — 1,115 mẫu)

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Random Forest** | **97.49%** | **89.78%** | 99.19% | 82.00% |
| Decision Tree | 96.59% | 87.42% | 86.84% | 88.00% |
| Naive Bayes | 96.14% | 83.27% | **100.0%** | 71.33% |
| Logistic Regression | 96.14% | 83.52% | 98.20% | 72.67% |

---

## 🚀 Cài đặt & Chạy

```bash
cd phishing_sms/code
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
```

Mở trình duyệt: **http://localhost:5000**

---

## 🌐 REST API

**`POST /api/predict`**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sms": "You won a FREE prize!", "model": "random_forest"}'
```

**`GET /api/metrics`** — trả về metrics tất cả model

---

## 🏗️ Cấu trúc

```
phishing_sms-main/
├── README.md
└── code/
    ├── app.py
    ├── model_random_forest.pkl       # 97.49%
    ├── model_decision_tree.pkl
    ├── model_naive_bayes.pkl
    ├── model_logistic_regression.pkl
    ├── vectorizer_sms.pkl
    ├── requirements.txt
    └── templates/
        ├── index.html
        └── metrics.html
```