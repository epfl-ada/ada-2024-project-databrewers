
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
from sklearn.feature_extraction.text import CountVectorizer

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

    reviews = reviews.dropna(subset=['aroma', 'palate'])
    reviews = reviews[(reviews['aroma'] + reviews['palate']) >= 8.0]

    flavour_counts = process_flavours(reviews, all_flavours)
    

    reviews = pd.concat([reviews, flavour_counts], axis=1)
    

    reviews = reviews[flavour_counts.sum(axis=1) > 0]

    total_flavour_mentions_per_month = flavour_counts.groupby(reviews['month']).transform('sum').sum(axis=1)
    reviews['total_flavour_mentions'] = total_flavour_mentions_per_month
    
    print("Normalizing flavour occurrences by total flavour mentions per month...")

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
    

    plot_data['flavour'] = plot_data['flavour'].str.replace('_normalized', '', regex=False)
    

    if not pd.api.types.is_datetime64_any_dtype(plot_data['month']):
        plot_data['month'] = pd.to_datetime(plot_data['month'])
    

    plot_data = plot_data.sort_values('month')

    primary_plot_data = plot_data[plot_data['flavour'].isin(primary_flavours)]
    other_plot_data = plot_data[plot_data['flavour'].isin(other_flavours)]

    fig, axes = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
    
    # Plot for Primary Flavours
    sns.lineplot(
        data=primary_plot_data, 
        x='month', 
        y='normalized_occurrence', 
        hue='flavour', 
        marker='o', 
        ax=axes
    )
    axes.set_title('Normalized Occurrences of Citrus, Sweet, and Bitter Flavours Over Time', fontsize=20)
    axes.set_xlabel('Month', fontsize=18)
    axes.set_ylabel('Normalized Occurrences', fontsize=18)
    axes.legend(title='Flavour', fontsize=16)
    
    
    # Improve layout
    plt.tight_layout()
    plt.show()
    
    return reviews

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def group_styles_by_flavours(reviews):
    """
    Groups beer reviews by style, normalizes flavour mentions, and plots the distribution.

    Parameters:
    - reviews (pd.DataFrame): DataFrame containing beer reviews with a 'style_simp' column 
      and flavour columns ['hoppy', 'malty', 'fruity', 'spicy', 'citrus', 
      'sweet', 'bitter', 'sour', 'tart', 'crisp'].

    Returns:
    - pd.DataFrame: DataFrame with normalized flavour percentages per style.
    """
    # Define the list of flavours
    flavours = ['hoppy', 'malty', 'fruity', 'spicy', 'citrus', 
                    'sweet', 'bitter', 'sour', 'tart', 'crisp']
    
    main_styles =  ['IPA', 'Stout', 'Pilsner', 'Porter', 'Lager']
    

    missing_flavours = [flavour for flavour in flavours if flavour not in reviews.columns]
    if missing_flavours:
        raise ValueError(f"The following flavour columns are missing in the DataFrame: {missing_flavours}")

    style_flavours = reviews.groupby('style_simp')[flavours].sum()

    print("Calculating total flavour mentions per style...")
    style_flavours['total_flavours'] = style_flavours.sum(axis=1)
    style_flavours['total_flavours'].replace(0, pd.NA, inplace=True)
    normalized_flavours = style_flavours[flavours].div(style_flavours['total_flavours'], axis=0) * 100
    normalized_flavours.dropna(inplace=True)
    
    normalized_flavours = normalized_flavours.reset_index()
    normalized_flavours = normalized_flavours[normalized_flavours['style_simp'].isin(main_styles)]
    
    
    plot_data = normalized_flavours.melt(
        id_vars='style_simp', 
        value_vars=flavours, 
        var_name='Flavour', 
        value_name='Percentage'
    )
    
    # Initialize the matplotlib figure
    plt.figure(figsize=(18, 12))
    
    # Create a bar plot with 'style_simp' on the x-axis and 'Percentage' on the y-axis
    sns.barplot(
        data=plot_data, 
        x='style_simp', 
        y='Percentage', 
        hue='Flavour',
        palette='Set2'  # Choose a color palette for better distinction
    )
    
    # Set plot titles and labels
    plt.title('Normalized Flavour Occurrences in US Beer Reviews by Style', fontsize=20)
    plt.xlabel('Beer Style', fontsize=18)
    plt.ylabel('Percentage of Flavour Mentions (%)', fontsize=18)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=18)
    
    # Adjust legend
    plt.legend(title='Flavour', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    
    # Improve layout to prevent clipping of labels and legend
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
    return normalized_flavours


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
    
# Function to preprocess and extract words related to the attributes from reviews
def preprocess_text_for_attribute(reviews, related_words):
    all_words = []
    for review in reviews['cleaned_tokens']:  
        for word in review:
            if word in related_words:
                all_words.append(word)
    return all_words

def generate_season_wordclouds(season_name, season_reviews, aroma_words_list, palate_words_list, mouthfeel_words_list, taste_words_list):
    """
    Generates a 2x2 subplot with word clouds for aroma, palate, mouthfeel, and taste for a given season.

    Parameters:
    - season_name: The name of the season (e.g., "Winter", "Spring").
    - season_reviews: A dictionary containing reviews for each season.
    - aroma_words_list: List of words associated with aroma.
    - palate_words_list: List of words associated with palate.
    - mouthfeel_words_list: List of words associated with mouthfeel.
    - taste_words_list: List of words associated with taste.
    """
    
    # Preprocess reviews for each characteristic
    palate_words = preprocess_text_for_attribute(season_reviews, palate_words_list)
    aroma_words = preprocess_text_for_attribute(season_reviews, aroma_words_list)
    mouthfeel_words = preprocess_text_for_attribute(season_reviews, mouthfeel_words_list)
    taste_words = preprocess_text_for_attribute(season_reviews, taste_words_list)

    words_data = {
        "Palate": palate_words,
        "Aroma": aroma_words,
        "Mouthfeel": mouthfeel_words,
        "Taste": taste_words
    }

    # Generate 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Word clouds for beer characteristics: {season_name}", fontsize=16)

    for ax, (attribute, words) in zip(axes.flat, words_data.items()):
        word_freq = Counter(words)
        if len(word_freq) > 0:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(attribute, fontsize=12)
        else:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()

    