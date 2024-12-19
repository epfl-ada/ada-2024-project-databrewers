import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns

# Helper function to preprocess text
def preprocess_text(text):
    return re.findall(r'\b\w+\b', text.lower()) 


def clean_data(reviews):
    """
    Cleans the data by removing rows where 'text' is NaN or not a string, and applying the preprocess_text function

    Args:
    reviews: DataFrame containing reviews

    Returns:
    reviews_clean: DataFrame containing cleaned reviews
    """

    # Remove rows where 'text' is NaN or not a string
    reviews_clean = reviews[reviews['text'].apply(lambda x: isinstance(x, str))]
    # Now apply the preprocess_text function
    reviews_clean['cleaned_tokens'] = reviews_clean['text'].apply(preprocess_text)

    return reviews_clean


def get_words(reviews_high, reviews_low):
    """
    
    Gets words from reviews with high and low ratings

    Args:
    reviews_high: DataFrame containing reviews with high ratings
    reviews_low: DataFrame containing reviews with low ratings

    Returns:
    all_words_high: List of words in reviews with high ratings
    all_words_low: List of words in reviews with low ratings
    """

    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Generate bigrams for each group, excluding stop words and non-alphabetic tokens
    bigrams_high = [
        " ".join(bigram)
        for tokens in tqdm(reviews_high['cleaned_tokens'], desc="Processing High Ratings")
        for bigram in zip(tokens[:-1], tokens[1:])  # Create bigrams
        if all(
            word.isalpha() and word.lower() not in stop_words and word.lower() != "beer"
            for word in bigram
        )  # Filter out stop words and specific unwanted words
    ]

    bigrams_low = [
        " ".join(bigram)
        for tokens in tqdm(reviews_low['cleaned_tokens'], desc="Processing Low Ratings")
        for bigram in zip(tokens[:-1], tokens[1:])  # Create bigrams
        if all(
            word.isalpha() and word.lower() not in stop_words and word.lower() != "beer"
            for word in bigram
        )  # Filter out stop words and specific unwanted words
    ]

    return bigrams_high, bigrams_low


def plot_word_clouds(wordcloud_high, wordcloud_low):
    """
    Plots word clouds for reviews with high and low ratings.

    Args:
    wordcloud_high: WordCloud object for high ratings
    wordcloud_low: WordCloud object for low ratings
    """

    # Plot the word clouds
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_high, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most common pair of words used in beer reviews with a rating higher than 4')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_low, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most common pair of words used in beer reviews with a rating lower than 3')

    plt.tight_layout()
    plt.show()


# Helper function to calculate the percentage of reviews containing each bigram
def calculate_percentage_of_reviews_containing_bigram(bigrams, reviews):
    percentages = []

    for bigram, _ in bigrams:
        # Check how many reviews contain this bigram
        num_reviews_containing_bigram = sum(
            1 for tokens in reviews['cleaned_tokens'] if tuple(bigram.split()) in zip(tokens[:-1], tokens[1:])
        )
        percentage = (num_reviews_containing_bigram / len(reviews)) * 100 if len(reviews) > 0 else 0
        percentages.append((bigram, percentage))

    return percentages

def compare_high_low(bigram_freq_high, bigram_freq_low, reviews_high, reviews_low):
    """
    Compare the most common bigrams in reviews with high and low ratings.

    Args:
    bigram_freq_high: Counter object containing bigram frequencies for reviews with high ratings
    bigram_freq_low: Counter object containing bigram frequencies for reviews with low ratings
    reviews_high: DataFrame containing reviews with high ratings
    reviews_low: DataFrame containing reviews with low ratings

    Returns:
    df_comparison: DataFrame comparing the most common bigrams in high and low
    """

    # Get the top 20 most common bigrams in each group
    top_bigrams_high = bigram_freq_high.most_common(20)
    top_bigrams_low = bigram_freq_low.most_common(20)

    # Calculate percentages for each group
    percentages_high = calculate_percentage_of_reviews_containing_bigram(top_bigrams_high, reviews_high)
    percentages_low = calculate_percentage_of_reviews_containing_bigram(top_bigrams_low, reviews_low)

    # Create DataFrames for comparison
    df_high = pd.DataFrame(percentages_high, columns=['Bigram', 'Percentage High Reviews'])
    df_low = pd.DataFrame(percentages_low, columns=['Bigram', 'Percentage Low Reviews'])

    # Merge the DataFrames
    df_comparison = pd.merge(df_high, df_low, on='Bigram', how='outer')

    # Sort by the most frequent bigrams in the high group
    df_comparison = df_comparison.sort_values(by='Percentage High Reviews', ascending=False)

    return df_comparison