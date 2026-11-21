# smart-comment-classifier
ML, NLP and LLM powered spam and toxicity detection system

Smart Comment Classifier

ML + NLP + LLM-Based Comment Moderation System

Overview

Smart Comment Classifier is an intelligent comment moderation system designed to detect spam and toxic content in user-generated comments.

The project combines Machine Learning, Natural Language Processing, Transformer-based models, and Large Language Models (LLMs) to assist in identifying potentially harmful or unwanted comments.

Key Features

- Spam Detection: Identifies spam and legitimate comments.
- Toxicity Detection: Detects toxic language and harmful content.
- Confidence Scores: Provides prediction confidence.
- LLM-Assisted Explanations: Generates explanations for moderation predictions.
- Safer Text Rewriting: Suggests alternative wording for toxic comments.
- Batch Processing: Supports CSV-based comment analysis.
- Human Review: Identifies comments that may require manual moderation.

Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- PyTorch
- Hugging Face Transformers
- DistilBERT
- FLAN-T5
- Streamlit
- Google Colab

Project Architecture

             User Comments
                   |
                   v
           Text Preprocessing
                   |
          +--------+--------+
          |                 |
          v                 v
     Spam Detection    Toxicity Detection
          |                 |
          +--------+--------+
                   |
                   v
           Prediction Results
                   |
                   v
           LLM-Assisted Layer
                   |
          +--------+--------+
          |                 |
          v                 v
      Explanation      Safer Rewrite
          |                 |
          +--------+--------+
                   |
                   v
           Streamlit Dashboard

Repository Structure

smart-comment-classifier/
│
├── notebooks/
│   └── Model development and experiments
│
├── streamlit_app/
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md

Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Class-wise performance

The project also considers false positives and false negatives to understand moderation errors.

Installation and Usage

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run streamlit_app/app.py

Note: Trained model files and required configuration files must be available at the paths expected by the application.

Responsible AI

- Predictions are model estimates, not absolute judgments.
- Uncertain comments should be reviewed by humans.
- LLM explanations should not override classifier predictions.
- The system should be tested for false positives, false negatives, and language bias.

Future Improvements

- Improve generalization on unseen comments.
- Add multilingual and Hinglish support.
- Improve confidence calibration.
- Add moderation analytics.
- Explore API integration and production deployment.

Project Status

Model development, evaluation, LLM integration, and Streamlit application development are part of the project. Deployment and end-to-end validation are ongoing.

Author

Deepak Kumar

B.Tech – Computer Science and Engineering

GitHub: https://github.com/YOUR_USERNAME
