# MovieReview: Dual-approach Sentiment Analysis for Movie Reviews

Welcome to **MovieReview**, a powerful tool for analyzing movie review sentiment using two distinct machine learning approaches: **Logistic Regression** (traditional ML) and **LSTM Neural Networks** (deep learning). This project provides a complete ML pipeline—from preprocessing to deployment—delivered through an intuitive web interface for real-time predictions. Ideal for filmmakers, critics, and data enthusiasts!

## Overview

MovieReview leverages a dual-approach sentiment analysis system to evaluate movie reviews with high accuracy. Whether you prefer the simplicity of traditional machine learning or the depth of neural networks, this tool offers flexibility and robust performance. The project uses the IMDB dataset and includes a Flask-based web application for seamless interaction.

## Features

- **Dual Models**: Choose between Logistic Regression (traditional ML) and LSTM Network (deep learning) for sentiment analysis.
- **Real-time Predictions**: Analyze reviews instantly via the web interface.
- **Complete ML Pipeline**: Includes data preprocessing, model training, and deployment.
- **User-friendly**: Simple interface for filmmakers, critics, and researchers.
- **Open Source**: Contribute and customize to suit your needs!

## Model Performance Metrics

The following metrics are based on the evaluation of the models using the IMDB dataset (train-test split: 80-20). These values are indicative and may vary depending on the specific dataset preprocessing and training conditions. Last updated: 11:44 AM PKT, Saturday, November 01, 2025.

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 85.2%    | 84.7%     | 86.0%  | 85.3%    |
| LSTM Network       | 88.7%    | 87.9%     | 89.5%  | 88.7%    |

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were identified correctly.
- **F1-Score**: Harmonic mean of precision and recall, providing a balanced measure.

*Note*: These metrics are placeholders. For exact values, evaluate your trained models and update this section accordingly.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/moviereview.git
   pip install -r requirements.txt
   
