import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# Load the dataset
data = pd.read_csv('Train_data.csv')

# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 0'])

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])

# Save the LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label_encoded'], test_size=0.2, random_state=42
)

# Save preprocessed data to a CSV file
preprocessed_data = pd.DataFrame({
    'text': X_train.tolist() + X_test.tolist(),
    'label': y_train.tolist() + y_test.tolist()
})
preprocessed_data.to_csv('Preprocessed_dataa.csv', index=False)

print("Preprocessed data saved to preprocessed_data.csv")
