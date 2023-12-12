# Standard library imports
import os
import re
import zipfile
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Third-party imports
import numpy as np
from numpy import zeros
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inflect
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    log_loss, 
    f1_score, 
    roc_auc_score, 
    roc_curve
)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import (
    Dense, 
    Input, 
    Dropout, 
    Flatten, 
    concatenate, 
    LSTM, 
    Conv1D, 
    BatchNormalization, 
    Bidirectional, 
    Embedding
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils import np_utils

# Special utility imports
#from google.colab import files

# Download necessary NLTK data
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Set global settings
#%matplotlib inline
p = inflect.engine()
STOPWORDS = set(stopwords.words('english'))

def separate_features_and_target(train: pd.DataFrame, test: pd.DataFrame, target_col: str):
    """
    Separates features and target variables for training and testing sets.

    Parameters:
    train (pd.DataFrame): The training data set.
    test (pd.DataFrame): The testing data set.
    target_col (str): The name of the target variable column in the DataFrame.

    Returns:
    tuple: Returns four DataFrames:
           - X_train: Feature variables for the training set.
           - y_train: Target variable for the training set.
           - X_test: Feature variables for the testing set.
           - y_test: Target variable for the testing set.
    """

    # Extract target column ('Rating') from the training set
    y_train = train[target_col].copy()
    # Drop the target column from the feature set for training
    X_train = train.drop(target_col, axis=1)

    # Extract target column ('Rating') from the testing set
    y_test = test[target_col].copy()
    # Drop the target column from the feature set for testing
    X_test = test.drop(target_col, axis=1)

    return X_train, y_train, X_test, y_test

def tokenize_reviews(X_train, X_test, tokenizer):
    """
    Tokenizes the "Reviews" column in the training and testing DataFrames and adds the tokenized sequences as a new column.

    Parameters:
        X_train (DataFrame): The training dataset containing a "Reviews" column with text data to tokenize.
        X_test (DataFrame): The testing dataset containing a "Reviews" column with text data to tokenize.
        tokenizer (Tokenizer): The fitted Keras Tokenizer object.

    Returns:
        X_train (DataFrame): The training dataset with the new "text_tokenizer" column containing tokenized sequences.
        X_test (DataFrame): The testing dataset with the new "text_tokenizer" column containing tokenized sequences.
    """

    # Tokenize the "Reviews" column in the training dataset and store it in a new column called "text_tokenizer"
    X_train['text_tokenizer'] = tokenizer.texts_to_sequences(X_train['Reviews'].values)

    # Tokenize the "Reviews" column in the testing dataset and store it in a new column called "text_tokenizer"
    X_test['text_tokenizer'] = tokenizer.texts_to_sequences(X_test['Reviews'].values)

    return X_train, X_test

def pad_text_sequences(X_train: pd.DataFrame, X_test: pd.DataFrame, maxlen: int = 1000) -> (pd.DataFrame, pd.DataFrame):
    """
    Pads the text token sequences in the 'text_tokenizer' column of both training and testing sets to a fixed length.

    Parameters:
    X_train (pd.DataFrame): The feature variables for the training set with a 'text_tokenizer' column.
    X_test (pd.DataFrame): The feature variables for the testing set with a 'text_tokenizer' column.
    maxlen (int): The maximum length to which the sequences will be padded.

    Returns:
    tuple: Returns two NumPy arrays:
           - X_train_pad: The padded sequences for the training set.
           - X_test_pad: The padded sequences for the testing set.
    """

    # Pad the token sequences in the 'text_tokenizer' column of the training set
    X_train_pad = sequence.pad_sequences(X_train['text_tokenizer'].values, maxlen=maxlen, padding='post')

    # Pad the token sequences in the 'text_tokenizer' column of the testing set
    X_test_pad = sequence.pad_sequences(X_test['text_tokenizer'].values, maxlen=maxlen, padding='post')

    return X_train_pad, X_test_pad


def download_glove_embeddings(glove_url, glove_dir):
    """
    Downloads GloVe embeddings if they are not already present.
    
    Args:
    - glove_url (str): The URL where GloVe embeddings can be downloaded.
    - glove_dir (str): Local directory where the GloVe embeddings should be saved.
    
    Returns:
    None
    """
    
    # Create the directory if it doesn't exist
    if not os.path.exists(glove_dir):
        os.makedirs(glove_dir)

    # Define the path to check if the file is already downloaded
    glove_zip_path = os.path.join(glove_dir, 'glove.twitter.27B.zip')

    # Check if the file is already downloaded. If not, download it.
    if not os.path.exists(glove_zip_path):
        print("GloVe embeddings not found. Downloading...")
        urllib.request.urlretrieve(glove_url, glove_zip_path)
        print(f"Downloaded GloVe embeddings to {glove_zip_path}")
    else:
        print("GloVe embeddings already exist. Skipping download.")

# Define the URL and the local path where the file should be saved
#GLOVE_URL = 'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
#GLOVE_DIR = 'data/glove'

# Call the function
#download_glove_embeddings(GLOVE_URL, GLOVE_DIR)

def load_glove_embeddings(glove_zip_path: str):
    """
    Load GloVe vectors from a ZIP file into a Python dictionary.

    Parameters:
    glove_zip_path (str): The path to the ZIP file containing the pre-trained word vectors.

    Returns:
    tuple: Returns a dictionary and a set:
           - glove: A dictionary where keys are words and values are the corresponding GloVe vectors.
           - glove_words: A set containing the words that have GloVe vectors.
    """

    # Initialize an empty dictionary to store the GloVe vectors
    glove = {}

    # Open the ZIP file and extract the specific txt file
    with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
        with zip_ref.open('glove.twitter.27B.100d.txt') as f:
            # Read line by line from the extracted txt file
            for line in f:
                # Decode bytes to string
                line = line.decode('utf-8')
                # Split each line to get the word and its corresponding vector
                values = line.split()
                word = values[0]
                # Convert the vector values to a NumPy array of type float32
                vector = np.asarray(values[1:], dtype='float32')

                # Populate the dictionary with the word and its corresponding vector
                glove[word] = vector

    # Create a set containing the words that have GloVe vectors
    glove_words = set(glove.keys())

    return glove, glove_words

def create_embedding_matrix(tokenizer, glove_words, glove, embedding_dim=100):
    """
    Create an embedding matrix using pre-trained GloVe vectors.

    Parameters:
    tokenizer (Tokenizer): A trained Keras Tokenizer object.
    glove_words (set): A set containing words that have GloVe vectors.
    glove (dict): A dictionary where keys are words and values are their corresponding GloVe vectors.
    embedding_dim (int): The number of dimensions for the GloVe word vectors. Default is 100.

    Returns:
    np.ndarray: A NumPy array representing the embedding matrix.
    """

    # Get the vocabulary size from the tokenizer's word index
    max_vocabulary = len(tokenizer.word_index)

    # Initialize a matrix with zeros having shape (max_vocabulary+1, embedding_dim)
    embedding_matrix = zeros((max_vocabulary+1, embedding_dim))

    # Fill in the embedding matrix
    for word, i in tokenizer.word_index.items():
        # Check if the word exists in the GloVe word set
        if word in glove_words:
            # Get the GloVe vector for the word
            embedding_vector = glove[word]
            # Insert the GloVe vector in the embedding matrix at index i
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def LSTM_Glove_model(embedding_matrix, max_vocabulary):
    """
    Creates a Keras model for text classification based on reviews.

    Parameters:
        embedding_matrix (ndarray): Pre-trained embedding weights.
        max_vocabulary (int): Maximum number of unique words in the vocabulary.

    Returns:
        model (Model): The Keras Model object.
    """

    # Define the shape of the input for the review (1000 tokens)
    review = Input(shape=(1000,), name='review_input')

    # Embedding layer
    # Uses pre-trained weights (embedding_matrix) and sets it to non-trainable
    X = Embedding(output_dim=100,
                  input_dim=max_vocabulary + 1,
                  input_length=1460,
                  weights=[embedding_matrix],
                  trainable=False)(review)

    # LSTM layer
    # Uses Bidirectional LSTM with 100 units
    lstm_review = Bidirectional(LSTM(100))(X)

    # Dropout layer to reduce overfitting
    model = Dropout(0.5)(lstm_review)

    # Flatten layer to prepare for dense layers
    model = Flatten()(model)

    # First Dense layer with 64 units, ReLU activation, and L2 regularization
    model = Dense(64, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001))(model)

    # Second Dense layer with 8 units, ReLU activation, and L2 regularization
    model = Dense(8, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001))(model)

    # Output layer with 1 unit (binary classification) and Sigmoid activation
    output = Dense(1, activation='sigmoid', name='output')(model)

    # Compile the model
    model = Model(inputs=[review], outputs=[output])

    # Print a summary of the model's architecture
    print(model.summary())

    return model

def evaluate_model_performance(Y_test, preds):
    """
    Evaluate the performance of a classification model based on various metrics.

    Parameters:
    Y_test (array-like): Ground truth (correct) target values.
    preds (array-like): Estimated target values returned by a classifier.

    Returns:
    None: Prints the performance metrics.
    """

    # Convert predicted probabilities to binary outputs (0 or 1)
    binary_preds = 1 * (preds > 0.5)

    # Calculate Accuracy Score: True Predictions / Total Predictions
    accuracy = accuracy_score(Y_test, binary_preds)

    # Calculate F1 Score: Harmonic mean of Precision and Recall
    f1 = f1_score(Y_test, binary_preds)

    # Calculate ROC AUC Score: Area under the Receiver Operating Characteristic curve
    roc_auc = roc_auc_score(Y_test, preds)

    # Calculate Precision: True Positives / (True Positives + False Positives)
    precision = precision_score(Y_test, binary_preds)

    # Calculate the Confusion Matrix: [tn, fp, fn, tp]
    tn, fp, fn, tp = confusion_matrix(Y_test, binary_preds).ravel()

    # Calculate Sensitivity: True Positives / (True Positives + False Negatives)
    sensitivity = tp / (tp + fn)

    # Calculate Specificity: True Negatives / (True Negatives + False Positives)
    specificity = tn / (tn + fp)

    # Calculate Cross-Entropy Loss: Measures performance of a classification model
    cross_entropy = log_loss(Y_test, preds)

    # Print the calculated metrics
    print(f'Accuracy Score: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC Score: {roc_auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Cross-Entropy Loss: {cross_entropy:.4f}')

def make_rounded_predictions(model, X_test):
    """
    Make predictions using a trained model, round the predictions to the nearest integer,
    and then cast them to float.

    Parameters:
    model (model object): The trained machine learning model.
    X_test (array-like or DataFrame): The data to make predictions on.

    Returns:
    numpy.ndarray: An array of rounded, float-casted predictions."""
    # Make predictions using the provided model
    predictions = model.predict(X_test)

    # Round the predictions to the nearest integer
    rounded_predictions = predictions.round()

    # Cast the rounded predictions to float
    rounded_predictions = rounded_predictions.astype("float")

    return rounded_predictions



def lstm_plot_confusion_matrix(predictions, Y_test, save_path='TEST_model_results/LSTM_Glove_model_confusion_matrix.png'):
    """
    Plot a confusion matrix heatmap for the given predicted and actual labels.

    Parameters:
    predictions (array-like): An array-like object containing predicted labels.
    Y_test (array-like): An array-like object containing actual labels.
    save_path (str, optional): Path to save the confusion matrix plot. 
                               Defaults to 'TEST_model_results/LSTM_Glove_model_confusion_matrix.png'.

    Returns:
    None: This function only displays a heatmap of the confusion matrix.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(Y_test, predictions)

    # Display the confusion matrix
    print("The Confusion Matrix for the test set is:")
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the plot in the specified directory
    plt.savefig(save_path)


def calculate_predicted_and_actual(cm):
    """
    Calculate and print the total number of predicted 'yes' and 'no' labels,
    as well as the total number of actual positive and negative reviews.

    Parameters:
    cm (2x2 array-like): The confusion matrix, typically generated by sklearn's confusion_matrix().

    Returns:
    None: Prints the calculated numbers for Predicted_yes, Predicted_no, positive_reviews, and negative_reviews.
    """

    # Calculate the total number of predicted 'yes' and 'no' labels
    Predicted_yes = cm[0, 1] + cm[1, 1]
    Predicted_no = cm[0, 0] + cm[1, 0]

    # Print the total number of predicted 'yes' and 'no' labels
    print(f'Total number of Predicted "Yes": {Predicted_yes}')
    print(f'Total number of Predicted "No": {Predicted_no}')

    # Calculate the total number of actual positive and negative reviews
    positive_reviews = cm[1, 0] + cm[1, 1]
    negative_reviews = cm[0, 0] + cm[0, 1]

    # Print the total number of actual positive and negative reviews
    print(f'Total number of actual Positive Reviews: {positive_reviews}')
    print(f'Total number of actual Negative Reviews: {negative_reviews}')

def plot_loss(history, save_path='loss_plot.png'):
    """
    Plot the training and validation loss and save the plot.

    Parameters:
    - history (History): Training history object returned by Keras's fit method.
    - save_path (str, optional): Path where the plot will be saved. Defaults to 'loss_plot.png'.
    """

    # Extract loss values from the history object
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Extract the number of epochs trained
    epochs = range(1, len(train_loss) + 1)

    # Plotting training and validation loss
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
