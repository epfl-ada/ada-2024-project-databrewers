
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
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

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

def filter_positive_reviews(reviews_df, rating_threshold=4):
    return reviews_df[reviews_df['rating'] >= rating_threshold]

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

def process_flavours(text, flavours):
    """
    Process the 'cleaned_tokens' column to count occurrences of each flavour.
    
    Parameters:
    - text: pd.DataFrame containing the 'cleaned_tokens' column.
    - flavours: list of flavour strings to count.
    
    Returns:
    - pd.DataFrame with counts of each flavour.
    """
    # Initialize CountVectorizer with the specified flavours as the vocabulary
    vectorizer = CountVectorizer(vocabulary=flavours, binary=False)
    
    # If 'cleaned_tokens' are lists, join them into strings
    if text['cleaned_tokens'].dtype == 'object' and isinstance(text['cleaned_tokens'].iloc[0], list):
        text['cleaned_tokens'] = text['cleaned_tokens'].apply(lambda tokens: ' '.join(tokens))
    
    # Fit and transform the 'cleaned_tokens' to get counts
    counts = vectorizer.transform(text['cleaned_tokens'])
    
    # Convert the counts to a DataFrame
    counts_df = pd.DataFrame(counts.toarray(), columns=flavours, index=text.index)
    
    return counts_df

def analyse_flavours(reviews: pd.DataFrame):
    """
    Analyze flavour occurrences in beer reviews and plot normalized occurrences over time.
    
    Parameters:
    - reviews: pd.DataFrame containing beer reviews with columns ['aroma', 'palate', 'cleaned_tokens', 'month'].
    
    Returns:
    - pd.DataFrame with additional flavour count and normalization columns.
    """
    # Define all flavours
    all_flavours = ['hoppy', 'malty', 'fruity', 'spicy', 'citrus', 
                    'sweet', 'bitter', 'sour', 'tart', 'crisp']
    
    # Define primary and other flavours
    primary_flavours = ['citrus', 'sweet', 'bitter']
    other_flavours = [flavour for flavour in all_flavours if flavour not in primary_flavours]
    
    # Drop reviews with missing 'aroma' or 'palate'
    reviews = reviews.dropna(subset=['aroma', 'palate'])
    
    # Keep reviews where the sum of 'aroma' and 'palate' is >= 8 (i.e., average >= 4)
    reviews = reviews[(reviews['aroma'] + reviews['palate']) >= 8.0]
    
    # Process all flavours using the optimized process_flavours function
    flavour_counts = process_flavours(reviews, all_flavours)
    
    # Add flavour counts to the reviews DataFrame
    reviews = pd.concat([reviews, flavour_counts], axis=1)
    
    # Keep only reviews where the sum of all flavour counts is > 0
    reviews = reviews[flavour_counts.sum(axis=1) > 0]
    
    # This sums up all flavour counts for each month
    total_flavour_mentions_per_month = flavour_counts.groupby(reviews['month']).transform('sum').sum(axis=1)
    reviews['total_flavour_mentions'] = total_flavour_mentions_per_month
    
    print("Normalizing flavour occurrences by total flavour mentions per month...")
    
    # Normalize flavour counts by total_flavour_mentions
    normalized_columns = [f"{flavour}_normalized" for flavour in all_flavours]
    reviews[normalized_columns] = flavour_counts.div(reviews['total_flavour_mentions'], axis=0)
    
    print("Flavour occurrences normalized.")
    
    # Prepare data for plotting
    plot_data = reviews.melt(
        id_vars=['month'], 
        value_vars=normalized_columns,
        var_name='flavour', 
        value_name='normalized_occurrence'
    )
    
    # Clean the 'flavour' column by removing '_normalized'
    plot_data['flavour'] = plot_data['flavour'].str.replace('_normalized', '', regex=False)
    
    # Convert 'month' to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(plot_data['month']):
        plot_data['month'] = pd.to_datetime(plot_data['month'])
    
    # Sort by month for proper plotting
    plot_data = plot_data.sort_values('month')
    
    # Separate data into primary and other flavours
    primary_plot_data = plot_data[plot_data['flavour'].isin(primary_flavours)]
    other_plot_data = plot_data[plot_data['flavour'].isin(other_flavours)]
    
    # Set up the plotting environment with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 20), sharex=True)
    
    # Plot for Primary Flavours
    sns.lineplot(
        data=primary_plot_data, 
        x='month', 
        y='normalized_occurrence', 
        hue='flavour', 
        marker='o', 
        ax=axes[0]
    )
    axes[0].set_title('Normalized Occurrences of Citrus, Sweet, and Bitter Flavours Over Time')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Normalized Occurrences')
    axes[0].legend(title='Flavour')
    
    # Plot for Other Flavours
    sns.lineplot(
        data=other_plot_data, 
        x='month', 
        y='normalized_occurrence', 
        hue='flavour', 
        marker='o', 
        ax=axes[1]
    )
    axes[1].set_title('Normalized Occurrences of Other Flavours Over Time')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Normalized Occurrences')
    axes[1].legend(title='Flavour')
    
    # Improve layout
    plt.tight_layout()
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


# Visualizes the occurrences of a specific word for each time period.
def visualize_word_occurrences(word_counts, word):
    time_periods = list(word_counts.keys())
    counts = list(word_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(time_periods, counts, color='skyblue')
    plt.xlabel('Time period')
    plt.ylabel(f'Occurrences of "{word.capitalize()}" related terms')
    plt.title(f'Occurrences of terms related to "{word.capitalize()}" per time period')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Analyzes the occurrences of a specific word in reviews grouped by time periods.
def word_analysis_by_time_period(reviews_dict, word):
    word_counts = {}

    for time_period, reviews in reviews_dict.items():
        word_counts[time_period] = reviews['cleaned_tokens'].apply(lambda tokens: tokens.count(word)).sum()

    return word_counts

def analyze_and_visualize_group_occurrences(groups, reviews_by_season):
    num_groups = len(groups)
    num_cols = 2  
    num_rows = (num_groups + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    fig.tight_layout(pad=5.0)

    # Flatten the axes for easy iteration
    axes = axes.flatten()

    # Iterate through the groups and visualize their occurrences
    for idx, (group_name, word_list) in enumerate(groups.items()):
        group_occurrences = {}

        # Analyze occurrences for each word in the group
        for word in word_list:
            word_counts = word_analysis_by_time_period(reviews_by_season, word=word)
            for season, count in word_counts.items():
                group_occurrences[season] = group_occurrences.get(season, 0) + count

        # Normalize occurrences to percentages
        total_reviews_by_season = {
            season: len(reviews) for season, reviews in reviews_by_season.items()
        }
        group_percentages = {
            season: (group_occurrences.get(season, 0) / total_reviews_by_season[season]) * 100
            if total_reviews_by_season[season] > 0 else 0
            for season in reviews_by_season
        }

        # Plot the group's data as a bar chart
        ax = axes[idx]
        ax.bar(group_percentages.keys(), group_percentages.values(), color='skyblue')
        ax.set_title(f"Positive reviews mentioning {group_name} related terms by season [%]")
        ax.set_ylabel(f"Positive reviews mentioning {group_name} related terms [%]")
        ax.set_xlabel("Season")
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide any unused subplots
    for idx in range(len(groups), len(axes)):
        axes[idx].set_visible(False)

    plt.show()


# Function to preprocess and extract word counts for each group, then calculate percentages
def preprocess_and_count(reviews, related_words):
    group_word_counts = {group: 0 for group in related_words.keys()}  # Initialize word count for each group

    # Calculate the total number of reviews
    total_reviews = len(reviews)

    # Iterate through reviews
    for review in reviews['cleaned_tokens']:
        for group, words_list in related_words.items():
            for word in review:
                if word in words_list:
                    group_word_counts[group] += 1

    # Calculate the percentage for each group based on the total number of reviews
    group_percentages = {group: (count / total_reviews) * 100 for group, count in group_word_counts.items()}

    return group_percentages


# Creates a treemap based on word percentages for a specific category.
def create_treemap(word_percentages, category_name, season):
    # Convert the word percentages into a format suitable for a treemap
    treemap_data = [{"Category": category_name, "Subcategory": group, "Percentage": percentage}
                    for group, percentage in word_percentages.items()]

    title = f"Occurrences of {category_name} related terms in positive {season} reviews (Percentage)"

    # Create the treemap using Plotly
    fig = px.treemap(
        treemap_data,
        path=["Category", "Subcategory"],
        values="Percentage",
        title=title
    )

    fig.show()
    
    
    