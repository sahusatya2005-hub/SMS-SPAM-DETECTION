# SMS Spam Detection Dataset 

This project uses the **SMS Spam Collection Dataset** to train a machine learning model capable of detecting spam messages using Natural Language Processing (NLP).

## 📂 Dataset Description
The dataset contains labeled SMS messages, categorized as:
- **ham** — normal (non-spam) messages  
- **spam** — unsolicited or advertising messages  

Typical columns:
- `v1` — label (`ham` or `spam`)
- `v2` — SMS message text

The dataset may include additional unnamed columns, which are ignored.

## 🧪 Project Workflow
1. **Load the dataset**
2. **Clean and preprocess text**  
   - Lowercasing  
   - Removing punctuation, numbers, URLs  
   - Tokenizing  
   - Removing stopwords  
   - Lemmatizing  
3. **Vectorization using TF-IDF**
4. **Model Training**
   - Logistic Regression with GridSearchCV
5. **Evaluation**
   - Accuracy  
   - Precision/Recall/F1  
   - Confusion Matrix  
   - ROC AUC
6. **Model Saving**
   - Saves trained model to `spam_classifier.joblib`

## ▶️ How to Train the Model
```
python train_spam_model.py
```

## ▶️ Using the Saved Model
```python
from joblib import load
model = load("spam_classifier.joblib")
model.predict(["Congratulations! You've won a free prize!"])
```

## 📁 File Locations
- Notebook path: `/mnt/data/sms-spam-detection-with-nlp.ipynb`
- Output model: `/mnt/data/spam_classifier.joblib`

## 📘 Requirements
Install dependencies:
```
pip install pandas scikit-learn nltk joblib
```

Ensure NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
``

