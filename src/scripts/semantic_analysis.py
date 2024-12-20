
import re
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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

analyzer = SentimentIntensityAnalyzer()


def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())  
    return words

def get_cleaned_reviews(reviews):
    tqdm.pandas()
    # Remove rows where 'text' is NaN or not a string
    reviews_clean = reviews[reviews['text'].progress_apply(lambda x: isinstance(x, str))]

    # Now apply the preprocess_text function
    reviews_clean['cleaned_tokens'] = reviews_clean ['text'].progress_apply(preprocess_text)
    return reviews_clean

def top_n_words(reviews,n :int):
    nltk.download('stopwords')
    # Flatten the list of words from all reviews
    all_words = [word for review_us in tqdm(reviews['cleaned_tokens'],desc="Processing") for word in review_us]
    stop_words = set(stopwords.words('english'))
    words = [word for word in tqdm(all_words, desc="Processing") if word.isalpha() and word not in stop_words and word.lower() != "beer"]

    # Count word frequencies
    word_freq = Counter(words)
    top_20_words = word_freq.most_common(n)
    return top_20_words, word_freq

def filter_positive_reviews(reviews_df, rating_threshold=4):
    return reviews_df[reviews_df['rating'] >= rating_threshold]

def gen_wordcloud(word_freq):
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color= 'white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word cloud of the most common words used in the US beer reviews')
    plt.show()
    
def plot_words(top_20_words):
    words, counts = zip(*top_20_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 20 most frequent words in US beer reviews')
    plt.tight_layout()
    plt.show()
    
# Function to compute sentiment
def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)['compound']
    return sentiment

# Function to classify words based on sentiment
def classify_words(word):
    sentiment_score = analyzer.polarity_scores(word)['compound']
    if sentiment_score > 0:
        return word, 'positive'
    elif sentiment_score < 0:
        return word, 'negative'
    return None  # Neutral words are ignored

# Analyze sentiment in parallel
def process_sentiments(top_words):
    print("Classifying words...")
    positive_words = []
    negative_words = []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(classify_words, top_words), total=len(top_words), desc="Word Classification Progress"))
        
    # Filter out None values for neutral words and separate positive and negative words
    for result in results:
        if result:  # Skip None results
            word, sentiment = result
            if sentiment == 'positive':
                positive_words.append(word)
            elif sentiment == 'negative':
                negative_words.append(word)
    
    print("Word classification completed.")
    return positive_words, negative_words
    
def sentiment_analysis(word_freq):
    # Initialize VADER sentiment analyzer
    
    # Get the 5000 most common words
    top_5000_words = [word for word, count in word_freq.most_common(5000)]

    # Process sentiment classification for the top 50000 words
    positive_words, negative_words = process_sentiments(top_5000_words)

    # Count the number of positive and negative words among these 50000 most common words
    num_positive_words = len(set(top_5000_words).intersection(set(positive_words)))
    num_negative_words = len(set(top_5000_words).intersection(set(negative_words)))

    print(f"Number of positive words among the top 5000 words: {num_positive_words}")
    print(f"Number of negative words among the top 5000 words: {num_negative_words}")
    # Count word frequencies for positive and negative words
    positive_word_freq = Counter(positive_words)
    negative_word_freq = Counter(negative_words)
    gen_wordcloud(positive_word_freq)
    gen_wordcloud(negative_word_freq)
    

# count the number of times each flavour appears in the reviews for each flavour
def count_flavour_occurrences(text, flavour):
    count = text.count(flavour)
    if count > 0:
        return 1
    return count

# Define a function to process the reviews for each flavour
def process_flavours(text, flavours):
    return {
        flavour: text['cleaned_tokens'].progress_apply(count_flavour_occurrences, args=(flavour,))
        for flavour in flavours
    }

def analyse_flavours(reviews):
    flavours = ['hoppy']
    #flavours = ['hoppy', 'malty', 'fruity', 'spicy', 'citrus', 'sweet', 'bitter', 'sour', 'tart', 'crisp']
    #keep only reviews where (aroma + palate)/2 is greater than 4 and the sum of all flavours is greater than 0
    #reviews = reviews[(reviews['aroma'] + reviews['palate']) / 2 > 4]
    # Process all flavours without using a for loop
    reviews.update(process_flavours(reviews, flavours))
    print(reviews.head())
    #reviews = reviews[reviews[flavours].sum(axis=1) > 0]
    # Calculate the total number of reviews per month
    reviews['total_reviews'] = reviews.groupby('month')['cleaned_tokens'].transform('size')

    #  Normalize each flavor's occurrences by the total reviews for the month
    for flavour in flavours:
        reviews[f"{flavour}_normalized"] = reviews[flavour] / reviews['total_reviews']

    # Plot the normalized occurrences
    plt.figure(figsize=(15, 10))
    plt.title('Normalized Flavour Occurrences in US Beer Reviews by Month')
    for flavour in tqdm(flavours):
        sns.lineplot(data=reviews, x='month', y=f"{flavour}_normalized", label=flavour)
    plt.xlabel('Month')
    plt.ylabel('Normalized Occurrences')
    plt.legend()
    plt.show()
    return reviews



def calculate_normalized_word_percentage(reviews_dict, word_list):
    """
    Calculates the normalized percentage of occurrences of a list of words for each season.

    Args:
    - reviews_dict (dict): Dictionary where keys are seasons and values are DataFrames of reviews.
    - word_list (list): List of words to analyze.

    Returns:
    - dict: A dictionary with seasons as keys and percentage occurrences as values.
    """
    percentages = {}

    for season, reviews in reviews_dict.items():
        total_words = reviews['text'].str.split().str.len().sum()  # Total words in the season
        combined_count = 0

        for word in word_list:
            combined_count += reviews['text'].str.lower().str.count(rf'\b{word}\b').sum()

        # Normalize by total words for the season
        percentages[season] = (combined_count / total_words) * 100 if total_words > 0 else 0

    return percentages


def visualize_normalized_word_percentages(percentages, group_name):
    """
    Visualizes the normalized percentage occurrences of a group of words for each season.

    Args:
    - percentages (dict): Dictionary where keys are seasons and values are percentages.
    - group_name (str): Name of the group being analyzed.
    """
    time_periods = list(percentages.keys())
    values = list(percentages.values())

    plt.figure(figsize=(10, 6))
    plt.bar(time_periods, values, color='skyblue')
    plt.xlabel('Season')
    plt.ylabel(f'Normalized Percentage of "{group_name}" words')
    plt.title(f'Normalized Percentage of "{group_name}" words per season')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
    