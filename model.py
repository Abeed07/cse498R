import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the model and vectorizer

# Load the dataset
data = pd.read_csv('Train_data.csv')  # Replace with your actual dataset file path

# Drop unnecessary columns (example)
data = data.drop(columns=['Unnamed: 0'])  # Assuming 'Unnamed: 0' is an unnecessary column

# Encode labels
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])  # Assuming 'label' is the column with disease names

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label_encoded'], test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train an SVM model with a linear kernel
model = SVC(kernel='linear')
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy: {accuracy:.4f}")  # Print the accuracy

# Print the classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Optionally, save the model, TF-IDF vectorizer, and label encoder
joblib.dump(model, 'svm_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
