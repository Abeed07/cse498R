import joblib
import numpy as np

# Load the trained model, TF-IDF vectorizer, and label encoder
model = joblib.load('svm_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to get disease prediction
def predict_disease(input_text):
    # Preprocess and vectorize the input text
    input_tfidf = tfidf.transform([input_text])  # Transform the input text
    prediction = model.predict(input_tfidf)  # Get the predicted label
    predicted_label = prediction[0]  # Extract the predicted label
    disease = label_encoder.inverse_transform([predicted_label])[0]  # Convert label back to disease name
    return disease

# Simple chatbot loop
print("Hello! I am your health assistant. Tell me your symptoms, and I will try to predict your disease.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye! Stay healthy!")
        break
    else:
        disease = predict_disease(user_input)  # Predict disease based on input
        print(f"Bot: Based on your symptoms, I think you may have {disease}.")
