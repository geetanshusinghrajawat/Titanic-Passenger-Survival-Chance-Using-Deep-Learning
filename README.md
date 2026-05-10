# 🚢 Titanic Survival Prediction

An ANN-based binary classification model that predicts whether a passenger would have survived the Titanic — trained on a 1 million row dataset and deployed as an interactive Streamlit web app.

## 🚀 Live Demo
[**Try it here →**](https://will-you-survive-the-titanic-by-geetanshu.streamlit.app/)

## 📌 Overview

Enter passenger details (class, gender, fare, embarkation point, family size) and the app predicts survival probability in real time. The model uses a fully connected ANN with early stopping trained on large-scale Titanic data.

## 🛠️ How It Works

1. **Data** — Large-scale Titanic dataset (`huge_1M_titanic.csv`) with 1 million rows; dropped irrelevant columns (PassengerId, Name, Age, Ticket, Cabin)
2. **Preprocessing** — Embarkation codes mapped to full names; missing values dropped
3. **Feature Encoding** — LabelEncoder for Sex; OneHotEncoder for Embarked (3 stations); StandardScaler for numerical features (Pclass, SibSp, Parch, Fare)
4. **Split** — Train / Validation (80/20) / Test (20 rows holdout)
5. **Model** — Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
6. **Training** — Adam (lr=0.01), binary crossentropy loss, up to 100 epochs with EarlyStopping (patience=5, restores best weights)
7. **Deployment** — Streamlit app with form inputs; shows survival probability and plain-English outcome

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler) |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | ANN — Dense → Dense → Sigmoid |
| Serialization | Pickle |
| Deployment | Streamlit |

## 📁 Project Structure

```
├── project.ipynb         # Data preprocessing, model training & evaluation
├── predict.ipynb         # Inference notebook for manual testing
├── app.py                # Streamlit web app
├── model.h5              # Saved Keras model
├── label_encoder.pkl     # Serialized LabelEncoder (Sex)
├── onehot_encoder.pkl    # Serialized OneHotEncoder (Embarked)
└── scalar_encoder.pkl    # Serialized StandardScaler
```

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/geetanshusinghrajawat/your-repo-name
cd your-repo-name

# Install dependencies
pip install streamlit tensorflow scikit-learn pandas numpy

# Run the app
streamlit run app.py
```

## 📊 Dataset

Large-scale Titanic dataset with 1 million synthetic passenger records, extending the original Kaggle Titanic dataset for more robust model training.

## 👤 Author

**Geetanshu Singh Rajawat**  
[LinkedIn](https://www.linkedin.com/in/geetanshu-singh-rajawat/) | [GitHub](https://github.com/geetanshusinghrajawat)
