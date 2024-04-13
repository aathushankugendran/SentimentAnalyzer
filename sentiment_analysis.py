# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv("musical1.tsv", sep='\t')

# Define a function to preprocess the text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Initialize PorterStemmer for stemming
    stemmer = PorterStemmer()
    # Stem each token in the text
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Initialize WordNetLemmatizer for lemmatization
    lemmatizer = WordNetLemmatizer()
    # Lemmatize each stemmed token in the text
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    
    # Join the lemmatized tokens back into text
    return " ".join(lemmatized_tokens)

# Preprocess the 'Review' column using the defined function
df['cleaned_text'] = df['Review'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer to convert text into numerical vectors
vectorizer = TfidfVectorizer()
# Fit and transform the preprocessed text data into numerical vectors
X = vectorizer.fit_transform(df['cleaned_text'])  # Features

# Target variable
y = df['Score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest classifier
clf = RandomForestClassifier()
# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = clf.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Accuracy: The accuracy of 0.8 indicates that the classifier correctly predicted 80% of the instances in the test set.

#Precision: The precision of 0.7936507936507936 suggests that out of all instances predicted as positive (1), approximately
#79.37% were actually positive. This means the classifier is relatively good at avoiding false positives.

#Recall: The recall of 0.8771929824561403 implies that the classifier correctly identified approximately 87.72% of the actual 
# positive instances in the dataset. This indicates that the classifier has relatively good coverage of positive instances.

#F1 Score: The F1 score of 0.8333333333333334 is the harmonic mean of precision and recall. It provides a balance between 
# precision and recall. A higher F1 score indicates better overall performance of the classifier.

#Based on these outputs, it seems that the classifier performs reasonably well in predicting positive and negative scores for the 
# musical instrument reviews. However, further analysis may be needed to understand if there are specific areas where the classifier 
# could be improved, such as handling imbalanced classes or fine-tuning model parameters.