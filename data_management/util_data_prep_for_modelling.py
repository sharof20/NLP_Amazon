# Imports for dataframe operations
import pandas as pd

# Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Imports for path operations
import os

# Imports for text processing
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# If you haven't already downloaded the NLTK datasets:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def dataframe_summary(df: pd.DataFrame) -> None:
    """
    Display a summary of the provided DataFrame including:
    - First 5 rows
    - Shape of the DataFrame
    - Descriptive statistics for the 'Rating' column
    - Count of unique values for each feature
    - Value counts for the 'Rating' column

    Args:
    - df (pd.DataFrame): The input DataFrame for which the summary needs to be displayed.

    Returns:
    - None
    """

    # Display the first 5 rows of the DataFrame
    print("First 5 rows of the DataFrame:")
    print(df.head())
    print("\n")  # Print newline for better formatting

    # Display the shape of the DataFrame
    print(f"Shape of the DataFrame: {df.shape}")
    print("\n")  # Print newline for better formatting

    # Display descriptive statistics for the 'Rating' column
    print("Descriptive statistics for 'Rating':")
    print(df['Rating'].describe())
    print("\n")  # Print newline for better formatting

    # Display the count of missing values for each column
    print("Count of missing values for each column:")
    print(df.isna().sum())
    print("\n")  # Print newline for better formatting

    # Display the count of unique values for each feature
    print("Unique value counts for each feature:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"The feature '{col}' has {unique_count} distinct values.")
    print("\n")  # Print newline for better formatting

# Function to create and save the plot
def plot_review_counts(df, save_path='data_management/plots', file_name='count_of_reviews_by_stars.png'):
    """
    Generate and save a bar plot that represents the count of reviews sorted by stars (Rating).

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing a 'Rating' column with review stars.

    save_path: str
        Path to the folder where the plot should be saved.

    file_name: str
        Name of the PNG picture to be saved.

    Returns:
    -------
    None
    """

    # Create the figure with custom dimensions
    plt.figure(figsize=(12, 6))

    # Use Seaborn to generate a bar plot
    ax = sns.countplot(x='Rating', data=df, order=sorted(df['Rating'].unique()), palette='viridis')

    # Add informative labels and title
    ax.set_title('Frequency Distribution of Review Scores', fontsize=16)
    ax.set_xlabel('Review Stars', fontsize=14)
    ax.set_ylabel('Number of Reviews', fontsize=14)

    # Add grid lines for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the figure with the specified file name
    plt.savefig(f"{save_path}/{file_name}")

    # Show the plot
    plt.show()


def process_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'Rating' column of the DataFrame:
    - Values below 3 are set to 0
    - Values above 3 are set to 1
    - Values exactly 3 remain unchanged

    Args:
    - df (pd.DataFrame): The input DataFrame containing the 'Rating' column to process.

    Returns:
    - pd.DataFrame: A copy of the original DataFrame with the processed 'Rating' column.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Process the 'Rating' column based on the given conditions
    df_copy['Rating'] = np.where(
        df_copy['Rating'] < 3, 0,
        np.where(df_copy['Rating'] > 3, 1, df_copy['Rating'])
    )

    return df_copy

def undersample_majority(df: pd.DataFrame, column_name: str = 'Rating') -> pd.DataFrame:
    """
    Undersample the majority class in a binary classification DataFrame.

    This function balances the two classes by undersampling the majority class
    to match the number of samples in the minority class.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the binary classification data.
    - column_name (str): The column with the binary labels. Default is 'Rating'.

    Returns:
    - pd.DataFrame: A balanced DataFrame with equal number of samples for each class.
    """

    # Count the samples for each rating
    count_0 = sum(df[column_name] == 0)
    count_1 = sum(df[column_name] == 1)

    # Undersample the majority class
    if count_0 < count_1:
        df_1 = df[df[column_name] == 1].sample(count_0)
        df_0 = df[df[column_name] == 0]
    else:
        df_0 = df[df[column_name] == 0].sample(count_1)
        df_1 = df[df[column_name] == 1]

    # Concatenate the balanced datasets and return
    balanced_df = pd.concat([df_0, df_1])

    # Shuffle the entire dataset
    balanced_df = balanced_df.sample(frac=1)

    return balanced_df

def add_word_count(df: pd.DataFrame, text_column: str = 'Reviews', new_column: str = 'count_words') -> pd.DataFrame:
    """
    Add a new feature to the DataFrame that counts the number of words in a specified text column.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - text_column (str): The column containing the text data for which word count needs to be computed. Default is 'Reviews'.
    - new_column (str): The name of the new column where word count will be stored. Default is 'count_words'.

    Returns:
    - pd.DataFrame: A DataFrame with the new word count feature.
    - int: The maximum word count across all rows in the DataFrame.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Add the 'count_words' feature
    df_copy[new_column] = df_copy[text_column].apply(lambda x: len(str(x).split()))

    # Return the modified DataFrame and the maximum word count
    return df_copy, max(df_copy[new_column])

def plot_and_save_avg_words_vs_rating(df: pd.DataFrame, save_path: str = 'data_management/plots', filename: str = 'avg_words_vs_rating.png'):
    """
    Generate and save a bar plot representing the relationship between average word count
    in customer reviews and their corresponding ratings.

    Parameters:
    -----------
    df: pd.DataFrame
        A DataFrame containing the columns 'Rating' and 'count_words'.
        - 'Rating' should contain the rating values.
        - 'count_words' should contain the average number of words in reviews for each rating.

    save_path: str, optional (default is '/content/drive/MyDrive/ose final/data/model results')
        The directory where the plot will be saved.

    filename: str, optional (default is 'avg_words_vs_rating.png')
        The name of the saved plot.

    Returns:
    --------
    None

    """

    # Create the plot
    plt.figure(figsize=(10, 7))

    # Generate the bar plot using Seaborn
    sns.barplot(x='Rating', y='count_words', data=df, palette='viridis')

    # Add titles and labels
    plt.title('Average Word Count in Customer Reviews Relative to Rating', fontsize=16)
    plt.xlabel('Rating', fontsize=14)
    plt.ylabel('Average Number of Words', fontsize=14)

    # Add gridlines for easier readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    plt.savefig(os.path.join(save_path, filename))

    # Display the plot
    plt.show()


def remove_emoji(text):
    """
    Remove emojis from a given text string.

    Parameters:
    - text (str): The input string containing emojis.

    Returns:
    - str: The input string with emojis removed.
    """
    # Unicode patterns for various emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags = re.UNICODE)

    # Removing emojis from the text
    return emoji_pattern.sub(r'', text)

def text_Preprocessing(texts):
    """
    Preprocess a list of text strings by performing the following:
    - Lowercasing
    - Removing emails
    - Removing digits
    - Removing special characters
    - Removing stopwords
    - Lemmatization

    Parameters:
    - texts (list of str): List of text strings to preprocess.

    Returns:
    - list of str: List of preprocessed text strings.
    """
    # Convert to lowercase
    reviews = [text.lower() for text in texts]

    # Remove emails
    reviews = [re.sub(r'\S+@\S+', '', text) for text in reviews]

    # Remove digits
    reviews = [re.sub(r'\d+', '', text) for text in reviews]

    # Remove special characters
    reviews = [re.sub(r'[^\w\s]', '', text) for text in reviews]

    # Remove extra spaces
    reviews = [text.strip() for text in reviews]

    # Remove emojis
    reviews = [remove_emoji(text) for text in reviews]

    # Define stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    cleaned_reviews = []
    for review in reviews:
        tokens = [word for word in word_tokenize(review) if not word in stop_words]
        cleaned_reviews.append(" ".join(tokens))

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lem_reviews = []
    for review in cleaned_reviews:
        lem_reviews.append(" ".join(list(map(lemmatizer.lemmatize, word_tokenize(review)))))

    return lem_reviews

def add_word_count_after_preprocessing(df, review_column_name):
    """
    Adds a new column to a DataFrame containing the count of words in each review after preprocessing.

    Parameters:
    - df (DataFrame): The DataFrame containing the reviews.
    - review_column_name (str): The name of the column in the DataFrame that contains the reviews.

    Returns:
    - DataFrame: The original DataFrame with an additional column named 'count_words_after_preprocess'.
    """
    # Lambda function to count words in each review
    count_words_lambda = lambda x: len(str(x).split())

    # Create a new column in DataFrame to store word count after preprocessing
    new_column_name = 'count_words_after_preprocess'
    df[new_column_name] = df[review_column_name].apply(count_words_lambda)

    return df
