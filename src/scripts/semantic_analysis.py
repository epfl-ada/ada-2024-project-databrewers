
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
tqdm.pandas()


def preprocess_text(text):
    words = re.findall(r'\b\w+\b', text.lower())  
    return words

def get_cleaned_reviews(reviews):
    
    # Remove rows where 'text' is NaN or not a string
    reviews_clean = reviews[reviews['text'].apply(lambda x: isinstance(x, str))]

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


def gen_wordcloud(word_freq,Title="Word Cloud"):
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color= 'white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(Title)
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
    gen_wordcloud(positive_word_freq,Title="Positive Words")
    gen_wordcloud(negative_word_freq,Title="Negative Words")
    

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

def analyse_flavours(reviews: pd.DataFrame):
    flavours = ['hoppy', 'malty', 'fruity', 'spicy', 'citrus', 'sweet', 'bitter', 'sour', 'tart', 'crisp']
    #keep only reviews where (aroma + palate)/2 is greater than 4 and the sum of all flavours is greater than 0
    reviews = reviews.dropna(subset=['aroma', 'palate'])
    reviews = reviews[(reviews['aroma'] + reviews['palate'])  >= 8.0]
    # Process all flavours without using a for loop
    reviews = process_flavours(reviews, flavours)
    reviews = reviews[reviews[flavours].sum(axis=1) > 0]
    # Step 1: Calculate the total number of reviews per month
    reviews['total_reviews'] = reviews.groupby('month')['cleaned_tokens'].transform('size')

    # Step 2: Normalize each flavor's occurrences by the total reviews for the month
    print("Normalizing flavour occurrences...")
    reviews[[f"{flavour}_normalized" for flavour in flavours]] = reviews[flavours].div(reviews['total_reviews'], axis=0)
    print("Flavour occurrences normalized.")
    plot_data = reviews.melt(id_vars=['month'], value_vars=[f"{f}_normalized" for f in flavours],
                         var_name='flavour', value_name='normalized_occurrence')
    plot_data['flavour'] = plot_data['flavour'].str.replace('_normalized', '')


    # Plot
    plt.figure(figsize=(15, 10))
    sns.lineplot(data=plot_data, x='month', y='normalized_occurrence', hue='flavour')
    plt.title('Normalized Flavour Occurrences in US Beer Reviews by Month')
    plt.xlabel('Month')
    plt.ylabel('Normalized Occurrences')
    plt.legend(title='Flavour')
    plt.show()
    return reviews

def group_styles_by_flavours(reviews):
    flavours = ['hoppy', 'malty', 'fruity', 'spicy', 'citrus', 'sweet', 'bitter', 'sour', 'tart', 'crisp']
    # Group by style and sum the flavour columns
    style_flavours = reviews.groupby('simplified_styles')[flavours].sum()
    
    # Normalize the flavour occurrences for each style in regards to the total flavours for that style
    for flavour in flavours:
        style_flavours[f"{flavour}_normalized"] = style_flavours[flavour] / style_flavours[flavours].sum(axis=1)
        
    # Plot the normalized flavour occurrences for each style
    plt.figure(figsize=(15, 10))
    plt.title('Normalized Flavour Occurrences in US Beer Reviews by Style')
    for flavour in tqdm(flavours):
        sns.barplot(data=style_flavours, x='simplified_styles', y=f"{flavour}_normalized", label=flavour)
    plt.xlabel('Style')
    plt.ylabel('Normalized Occurrences')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
    
    