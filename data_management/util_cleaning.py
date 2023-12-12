import os
import re
import glob
import string
import pandas as pd
from langdetect import detect
from googletrans import Translator

def data_cleaning(input_directory: str, output_file: str):
    """
    Perform end-to-end data cleaning and preparation.
    
    This function merges multiple CSV files from a directory into a single DataFrame,
    processes it, removes duplicates, refines the ratings, cleans the reviews,
    filters them, separates English and non-English reviews, translates non-English reviews,
    and finally saves the cleaned DataFrame to a CSV file.
    
    Parameters:
        input_directory (str): Path to the directory containing the original CSV files.
        output_file (str): Path to save the cleaned, processed CSV file.
        
    Returns:
        None: The function performs all operations in-place and saves the result to a CSV file.
    """
    # Merge all CSV files from the specified directory into a single DataFrame
    df = merge_csv_files(input_directory)
    
    # Process the merged DataFrame by reordering columns and shuffling rows
    df = process_dataframe(df)

    # Remove duplicates and handle missing values
    df = removing_duplicates(df)

    # Refine the ratings by extracting relevant portions and filtering
    df = refine_ratings(df)

    # Clean the review text using a custom function to remove unwanted characters and words
    df.loc[:, 'Reviews'] = df['Reviews'].apply(review_cleaning)

    # Further filter reviews to remove any remaining anomalies or duplicates
    df = filter_reviews(df)

    # Identify and separate English and non-English reviews
    non_english_mask = df['Reviews'].apply(is_non_english)
    english_df = df[~non_english_mask]
    non_english_df = df[non_english_mask]
    
    # Translate non-English reviews to English using Google Translate API
    non_english_df = non_english_df.copy()
    non_english_df['Reviews'] = non_english_df['Reviews'].apply(translate_review)

    # Concatenate the English and translated non-English reviews to form the final DataFrame
    df = pd.concat([english_df, non_english_df], ignore_index=True)

    # Save the cleaned and processed DataFrame to a CSV file
    df.to_csv(output_file, index=False)


def merge_csv_files(input_directory: str) -> pd.DataFrame:
    """
    Merge all CSV files from the specified directory into a single DataFrame.
    
    Parameters:
    - input_directory (str): The path to the directory containing CSV files.
    
    Returns:
    - pd.DataFrame: A single DataFrame containing data from all CSV files. A new column 'Index' is added to indicate row indices.
    """
    
    # Get a list of all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
    dataframes = []

    # Loop through each CSV file, read it into a DataFrame, and store in the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    # Concatenate the list of DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Add a new 'Index' column to the merged DataFrame containing row indices
    merged_df['Index'] = merged_df.index
    
    return merged_df

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a merged DataFrame by concatenating reviews, dropping unnecessary columns, 
    reordering columns, and shuffling the dataset.
    
    Parameters:
    - merged_df (pd.DataFrame): The initial DataFrame to be processed.
    
    Returns:
    - pd.DataFrame: The processed DataFrame.
    """
    
    # Concatenate 'Review' and 'Review_title' columns and store in a new column 'Reviews'
    df['Reviews'] = df['Review'] + df['Review_title']

    # Drop unnecessary columns from merged_df
    df = df.drop(['Review', 'Review_title', 'Link', 'Name'], axis=1)

    # Reorder the columns of merged_df
    new_column_order = ['Index', 'Rating', 'Reviews']
    df = df[new_column_order]

    # Shuffle the entire dataset
    df = df.sample(frac=1)
    
    return df

def removing_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by:
    1. Dropping rows with null values.
    2. Removing rows where 'Reviews' or 'Rating' columns contain only spaces.
    3. Removing duplicate reviews, retaining only the first occurrence.
    
    Parameters:
    - df (pd.DataFrame): The initial DataFrame to be cleaned.
    
    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    """
    # Drop rows containing any null values from the DataFrame
    df.dropna(inplace=True)
    
    # Check the count of null values in each column (for user information)
    print(df.isnull().sum())
    
    # Identify rows where 'Reviews' or 'Rating' columns contain only spaces
    rows_with_spaces = df[df[['Reviews', 'Rating']].apply(lambda col: col.str.strip().str.len() == 0).any(axis=1)]
    
    # Remove rows with spaces from the DataFrame
    df = df[~df.index.isin(rows_with_spaces.index)]

    # Calculate the number of duplicate reviews in the dataset
    num_duplicates = df.duplicated('Reviews').sum()
    print("Number of duplicate values:", num_duplicates)

    # Remove duplicates from 'Reviews' column, retaining the first occurrence
    df_no_duplicates = df.drop_duplicates(subset='Reviews', keep='first')
    
    # Calculate the number of duplicate reviews after removing duplicates in the dataset (for user confirmation)
    num_duplicates_after = df_no_duplicates.duplicated('Reviews').sum()
    print("Number of duplicate values after removing duplicates:", num_duplicates_after)

    return df_no_duplicates

def refine_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refines the 'Rating' column of a DataFrame by:
    1. Extracting the numerical part from strings like "4.5 stars".
    2. Converting the 'Rating' values to integers.
    3. Removing rows where 'Rating' is 3 to focus on distinctly Positive and Negative ratings.
    
    Additionally, the function prints:
    - Data types of each column.
    - Distribution of unique values in 'Rating' column.
    - Last few rows of the DataFrame for inspection.
    
    Parameters:
    - df (pd.DataFrame): The initial DataFrame to be refined.
    
    Returns:
    - pd.DataFrame: The refined DataFrame.
    """
    
    # Extract the first part of 'Rating' column
    df['Rating'] = df['Rating'].apply(lambda x: x.split(" ")[0])

    # Convert the extracted 'Rating' value to an integer
    df['Rating'] = df['Rating'].apply(lambda x: int(float(x)))

    # Remove rows where the 'Rating' is 3
    df = df[df['Rating'] != 3]

    # Print desired details
    print("Data types of each column:\n", df.dtypes)
    print("Distribution of unique values in 'Rating' column:\n", df['Rating'].value_counts())
    print("Last few rows of the DataFrame:\n", df.tail())
    
    return df

def review_cleaning(text):
    """
    Clean the input text by performing a series of preprocessing steps.
    
    Parameters:
    - text (str): The text to be cleaned.
    
    Returns:
    - str: The cleaned text.
    
    Steps:
    1. Convert the text to lowercase.
    2. Remove text within square brackets.
    3. Remove URLs.
    4. Remove HTML tags.
    5. Remove punctuation.
    6. Remove newlines.
    7. Remove words containing numbers.
    """
    
    # Convert the text to lowercase
    text = str(text).lower()
    
    # Remove text within square brackets
    text = re.sub('\[.*?\]', '', text)
    
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove newlines
    text = re.sub('\n', '', text)
    
    # Remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)
    
    return text

def filter_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out rows from a DataFrame based on the following criteria:
    1. Rows where the 'Reviews' column contains only spaces.
    2. Duplicate reviews, retaining only the first occurrence.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'Reviews' column.
    
    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    
    # Find rows where 'Reviews' column contains only spaces
    rows_with_spaces = df[df['Reviews'].str.strip().str.len() == 0]

    # Drop rows with spaces
    df = df[~df.index.isin(rows_with_spaces.index)]
    
    # Drop duplicates in 'Reviews' column, retaining the first occurrence
    df = df.drop_duplicates(subset='Reviews', keep='first')
    
    return df

"""# Function to detect non-English reviews
def is_non_english(review):
    try:
        return detect(review) != 'en'
    except:
        print(f"Could not detect language for review: {review}")
        return False

#translator = Translator()

# Translate the reviews
def translate_review(review):
    translator = Translator()
    try:
        translated_review = translator.translate(review, src='auto', dest='en').text
        return translated_review
    except Exception as e:
        print(f"An error occurred: {e}")
        return review # Return original review if translation fails
"""

def is_non_english(review):
    """
    Detect if a review is written in a language other than English.

    Parameters:
        review (str): The text of the review to check.

    Returns:
        bool: True if the review is not in English, False otherwise.

    Note:
        The function uses the 'detect' function from the 'langdetect' library to identify the language.
    """
    try:
        # Use 'detect' to identify the language of the review.
        return detect(review) != 'en'
    except Exception as e:
        # Log any exceptions that occur during language detection.
        print(f"Could not detect language for review: {review}. Exception: {e}")
        return False


def translate_review(review):
    """
    Translate a review into English using the Google Translate API.

    Parameters:
        review (str): The text of the review to translate.

    Returns:
        str: The translated review if successful, otherwise the original review.

    Note:
        The function uses the 'Translator' class from the 'googletrans' library.
    """
    # Create a new Translator object.
    translator = Translator()
    try:
        # Use 'translate' to translate the review to English.
        translated_review = translator.translate(review, src='auto', dest='en').text
        return translated_review
    except Exception as e:
        # Log any exceptions that occur during translation.
        print(f"An error occurred: {e}")
        return review  # Return original review if translation fails.
