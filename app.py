# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os
import warnings
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data files
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

class TextPreprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        # Remove HTML
        text = BeautifulSoup(text, "html.parser").get_text()
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def preprocess_pipeline(self, text):
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text
    
class FeatureExtractor:
    def __init__(self, vectorizer, tokenizer):
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

class SentimentAnalyzer:
    def __init__(self):
        self.lr_model = None
        self.lstm_model = None
        self.preprocessor = None
        self.feature_extractor = None
        self.tokenizer = None
        self.max_sequence_length = 200
        self.is_loaded = False
    
    def load_models(self):
        """Load all trained models and preprocessing objects from models/ directory"""
        try:
            models_dir = 'models'
            
            # Load Logistic Regression model (with warning suppression)
            
            lr_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
            if os.path.exists(lr_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.lr_model = joblib.load(lr_path)
                print("✅ Logistic Regression model loaded")
            else:
                print(f"❌ Logistic Regression model not found at: {lr_path}")
                return False
            
            # Load LSTM model
     
            lstm_path = os.path.join(models_dir, 'lstm_model.keras')
            if os.path.exists(lstm_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
                print("✅ LSTM model loaded")
            else:
                print(f"❌ LSTM model not found at: {lstm_path}")
                return False
            
            # Load preprocessor

            preprocess_path = os.path.join(models_dir, 'preprocessor.pkl')
            if os.path.exists(preprocess_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                with open(preprocess_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                print("✅ Preprocessor loaded")
            else:
                print(f"❌ Preprocessor model not found at: {preprocess_path}")
                return False
            
            # Load feature extractor (TF-IDF vectorizer)

            feature_extractor_path = os.path.join(models_dir, 'feature_extractor.pkl')
            if os.path.exists(feature_extractor_path):
                with open(feature_extractor_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                self.feature_extractor = FeatureExtractor(vectorizer=vectorizer, tokenizer=None)
                print("✅ Feature extractor loaded")    
            else:
                print(f"❌ Feature extractor not found at: {feature_extractor_path}")
                return False
                            
            # Load tokenizer
            tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print("✅ Tokenizer loaded")
            else:
                print(f"❌ Tokenizer not found at: {tokenizer_path}")
                return False
            
            self.is_loaded = True
            print("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_sentiment_lr(self, text):
        """Predict sentiment using Logistic Regression"""
        if not self.is_loaded:
            return {'error': 'Models not loaded'}
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.preprocess_pipeline(text)
            
            # Transform using TF-IDF
            tfidf_features = self.feature_extractor.transform([cleaned_text])
            
            # Predict
            prediction = self.lr_model.predict(tfidf_features)[0]
            probability = self.lr_model.predict_proba(tfidf_features)[0]
            
            sentiment = "positive" if prediction == 1 else "negative"
            confidence = max(probability)
            
            return {
                'sentiment': sentiment,
                'confidence': float(confidence),
                'model': 'Logistic Regression'
            }
        except Exception as e:
            return {'error': f'LR prediction error: {str(e)}'}
    
    def predict_sentiment_lstm(self, text):
        """Predict sentiment using LSTM"""
        if not self.is_loaded:
            return {'error': 'Models not loaded'}
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.preprocess_pipeline(text)
            
            # Convert to sequences
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
            
            # Predict
            prediction_proba = self.lstm_model.predict(padded_sequence, verbose=0)[0][0]
            sentiment = "positive" if prediction_proba > 0.5 else "negative"
            confidence = prediction_proba if sentiment == "positive" else 1 - prediction_proba
            
            return {
                'sentiment': sentiment,
                'confidence': float(confidence),
                'model': 'LSTM'
            }
        except Exception as e:
            return {'error': f'LSTM prediction error: {str(e)}'}

# Initialize analyzer
analyzer = SentimentAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint to verify models are loaded"""
    status = "loaded" if analyzer.is_loaded else "not loaded"
    models_info = {
        'logistic_regression': analyzer.lr_model is not None,
        'lstm': analyzer.lstm_model is not None,
        'preprocessor': analyzer.preprocessor is not None,
        'feature_extractor': analyzer.feature_extractor is not None,
        'tokenizer': analyzer.tokenizer is not None
    }
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': analyzer.is_loaded,
        'models_status': models_info,
        'message': f'Models are {status}'
    })

@app.route('/models-status')
def models_status():
    """Detailed models status endpoint"""
    models_info = {
        'logistic_regression': {
            'loaded': analyzer.lr_model is not None,
            'type': str(type(analyzer.lr_model)) if analyzer.lr_model else 'Not loaded'
        },
        'lstm': {
            'loaded': analyzer.lstm_model is not None,
            'type': str(type(analyzer.lstm_model)) if analyzer.lstm_model else 'Not loaded'
        },
        'preprocessor': {
            'loaded': analyzer.preprocessor is not None,
            'type': str(type(analyzer.preprocessor)) if analyzer.preprocessor else 'Not loaded'
        },
        'feature_extractor': {
            'loaded': analyzer.feature_extractor is not None,
            'type': str(type(analyzer.feature_extractor)) if analyzer.feature_extractor else 'Not loaded'
        },
        'tokenizer': {
            'loaded': analyzer.tokenizer is not None,
            'type': str(type(analyzer.tokenizer)) if analyzer.tokenizer else 'Not loaded'
        }
    }
    
    return jsonify({
        'overall_status': 'loaded' if analyzer.is_loaded else 'not loaded',
        'models': models_info
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model', 'logistic_regression')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not analyzer.is_loaded:
            return jsonify({'error': 'Models are not loaded. Please check the server.'}), 503
        
        if model_type == 'logistic_regression':
            result = analyzer.predict_sentiment_lr(text)
        elif model_type == 'lstm':
            if analyzer.lstm_model is None:
                return jsonify({'error': 'LSTM model is not available'}), 503
            result = analyzer.predict_sentiment_lstm(text)
        else:
            return jsonify({'error': 'Invalid model type. Use "logistic_regression" or "lstm"'}), 400
        
        # Check if there was an error in prediction
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload-models', methods=['POST'])
def reload_models():
    """Endpoint to reload models without restarting the server"""
    try:
        success = analyzer.load_models()
        if success:
            return jsonify({'message': 'Models reloaded successfully'})
        else:
            return jsonify({'error': 'Failed to reload models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load models when the app starts
def load_models_on_startup():
    print("🚀 Starting model loading from 'models' directory...")
    print("📁 Looking for model files:")
    print("   - models/logistic_regression_model.pkl")
    print("   - models/lstm.keras (or .h5)")
    print("   - models/preprocessor.pkl")
    print("   - models/feature_extractor.pkl")
    print("   - models/tokenizer.pkl")
    
    success = analyzer.load_models()
    if success:
        print("🎉 Models loaded successfully and ready for predictions!")
    else:
        print("❌ Failed to load models. Please check if all model files exist in the 'models' directory.")

# Load models immediately when module is imported
load_models_on_startup()

if __name__ == '__main__':
    print("🚀 Starting MovieSent Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)