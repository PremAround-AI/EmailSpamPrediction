# EmailSpamPrediction


This project is a simple yet effective implementation of an Email Spam Detection system using Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier. It classifies email text as either Spam or Ham (Not Spam) based on the words in the message.

🧩 Table of Contents :

Introduction

Features

Technologies Used

Dataset

Project Workflow

Installation

Usage

Results

Future Work


📜 Introduction :

Email spam is a common problem that clutters inboxes and can even pose security risks. This project demonstrates how to build a spam detection model using:

✅ NLP text preprocessing

✅ Feature extraction with Bag of Words / CountVectorizer

✅ Classification using Multinomial Naive Bayes

The goal is to automatically classify incoming email messages as "Spam" or "Ham".

⭐ Features :

Data cleaning and preprocessing (removing punctuation, stopwords, etc.)

Text vectorization (CountVectorizer / Bag-of-Words)

Model training with Multinomial Naive Bayes

Model evaluation (accuracy, confusion matrix)

Predicting custom email samples

🛠️ Technologies Used :

Python

scikit-learn

pandas

NumPy

NLTK 

Google Colab

📂 Dataset

The project uses a labeled SMS or Email dataset with spam/ham annotations.

Example sources :

SMS Spam Collection

Custom email datasets

Example format :

Label	Message
ham	Hey, are we still meeting today?
spam	Congratulations! You've won a prize!

⚙️ Project Workflow :

1️⃣ Load the dataset
2️⃣ Preprocess text (lowercasing, removing stopwords, punctuation)
3️⃣ Vectorize text using CountVectorizer
4️⃣ Split data into training and testing sets
5️⃣ Train Multinomial Naive Bayes classifier
6️⃣ Evaluate model performance
7️⃣ Predict new/unseen samples

🚀 Installation

Clone this repository :

bash
Copy
Edit
git clone https://github.com/PremAround-AI/EmailPrediction.git

cd spam-ham-detection

Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt content:

nginx
Copy
Edit
pandas
numpy
scikit-learn
nltk
🧪 Usage:

Run the Jupyter Notebook or Python script to:

✅ Load and preprocess the dataset

✅ Train the model

✅ Evaluate accuracy and metrics

✅ Make predictions on custom email messages

Example (in notebook) :

python
Copy
Edit
model.predict(["Congratulations! You've won a free iPhone"])
# Output: ['spam']

📊 Results:

Typical accuracy: 98% on spam datasets

Confusion matrix and classification report included

Works well for short message spam detection

(Include your actual metrics if you want!)

🔭 Future Work :

Use TF-IDF instead of CountVectorizer

Experiment with other classifiers (Logistic Regression, SVM)

Add email header analysis

Deploy as a web app (Flask, Streamlit)


