import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def format_data(data, id_type="review_id", chunk_size=1600):
    """
    Processes the given data to convert it to a more readable format

    Parameters:
    - data (pandas.DataFrame): input dataframe that contains the 'info' column with key-value pairs
    - id_type (str):  column name to be used for the unique identifier
    - chunk_size (int): size of each chunk to process at a time to optimize memory usage

    Returns:
    - processed_data (pandas.DataFrame): dataframe where each row corresponds to a unique ID
    """
    processed_data = []  # Temporary list to hold processed data

    for start in range(0, len(data), chunk_size):
        # Extract a chunk of data
        chunk = data.iloc[start:start + chunk_size].copy()

        # Add id_type based on the 16-row structure
        chunk[id_type] = chunk.index // 16

        # Split key-value pairs
        split_data = chunk['info'].str.split(': ', n=1, expand=True)
        chunk = chunk[split_data[1].notna()]  # Keep rows with valid key-value pairs

        # Assign key and value columns
        chunk[['key', 'value']] = split_data

        # Pivot the chunk to convert key to columns
        chunk_pivoted = chunk.pivot(index=id_type, columns='key', values='value').reset_index(drop=True)
        processed_data.append(chunk_pivoted)

    processed_data = pd.concat(processed_data, ignore_index=True)

    return processed_data



def seasonal_region_abv_test(reviews, abv_category, rating_column):
    """
    Perform ANOVA to assess if there are significant differences in ratings between regions (South, Midwest, Northeast, and West)
    across seasons for a specified ABV category (either 'low', 'medium', or 'high')and rating column (either aroma, palate, taste, appearance, overall, or rating). If the ANOVA test is significant, it runs Tukey's HSD test.

    Parameters:
    - reviews (pd.DataFrame): dataset containing reviews
    - abv_category (str): ABV category to filter by
    - rating_column (str): rating column to analyze

    Returns:
    - dict: Dictionary with seasons as keys and p-values/Tukey's HSD results as values
    """

    filtered_reviews = reviews[reviews['abv_category'] == abv_category]

    results = {}
    for season in reviews['season'].unique() \
            :

        season_data = filtered_reviews[filtered_reviews['season'] == season]

        ratings_by_region = [season_data[season_data['region'] == region][rating_column]
                             for region in season_data['region'].unique()]

        f_stat, p_value = stats.f_oneway(*ratings_by_region)
        results[season] = {'ANOVA_p_value': p_value}

        # If ANOVA is significant: perform Tukey's HSD
        if p_value < 0.05:
            tukey = pairwise_tukeyhsd(endog=season_data[rating_column], groups=season_data['region'], alpha=0.05)
            results[season]['Tukey_HSD'] = tukey.summary()
            print(f"season: {season}, ABV category: {abv_category}, rating column: {rating_column}")
            print(f"ANOVA p-value: {p_value:.4f} - Significant difference between regions")
            print(tukey.summary())
        else:
            print(f"season: {season}, ABV category: {abv_category}, rating column: {rating_column}")
            print(f"ANOVA p-value: {p_value:.4f} - No significant difference between regions")

        print("-" * 50)

    return results



def seasonal_region_test(reviews, rating_column):
    """
    Perform ANOVA to assess if there are significant differences in ratings between regions (South, Midwest, Northeast, and West)
    across seasons for a specified rating column (either aroma, palate, taste, appearance, overall, or rating).
    If the ANOVA test is significant, it runs Tukey's HSD test.

    Parameters:
    - reviews (pd.DataFrame): dataset containing reviews
    - rating_column (str): rating column to analyze

    Returns:
    - dict: Dictionary with seasons as keys and p-values/Tukey's HSD results as values
    """

    results = {}
    for season in reviews['season'].unique():

        season_data = reviews[reviews['season'] == season]

        # Group ratings by region
        ratings_by_region = [season_data[season_data['region'] == region][rating_column]
                             for region in season_data['region'].unique()]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*ratings_by_region)
        results[season] = {'ANOVA_p_value': p_value}

        # If ANOVA is significant: perform Tukey's HSD
        if p_value < 0.05:
            tukey = pairwise_tukeyhsd(endog=season_data[rating_column], groups=season_data['region'], alpha=0.05)
            results[season]['Tukey_HSD'] = tukey.summary()
            print(f"Season: {season}, Rating Column: {rating_column}")
            print(f"ANOVA p-value: {p_value:.4f} - Significant difference between regions")
            print(tukey.summary())
        else:
            print(f"Season: {season}, Rating Column: {rating_column}")
            print(f"ANOVA p-value: {p_value:.4f} - No significant difference between regions")

        print("-" * 50)

    return results