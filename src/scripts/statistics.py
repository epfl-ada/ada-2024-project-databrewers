import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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


def anova_test(review, rating_column, timescale, category):
    """
    Perform ANOVA to assess if there are significant differences in ratings between a given timescale
    for a given category across a rating column (either aroma, palate, taste, appearance, overall, or rating). 
    If the ANOVA test is significant, it runs Tukey's HSD test.

    Parameters:
    - reviews (pd.DataFrame): dataset containing reviews
    - rating_column (str): rating column to analyze
    - timescale (str): timescale to compare
    - category (str): category to analyze

    Returns:
    - dict: Dictionary with seasons as keys and p-values/Tukey's HSD results as values
    """


    results = {}
    for cat in review[category].unique() \
            :

        data = review[review[category] == cat]

        ratings = [data[data[timescale] == time][rating_column]
                             for time in data[timescale].unique()]

        f_stat, p_value = stats.f_oneway(*ratings)
        results[cat] = {'ANOVA_p_value': p_value}

        # If ANOVA is significant: perform Tukey's HSD
        if p_value < 0.05:
            tukey = pairwise_tukeyhsd(endog=data[rating_column], groups=data[timescale], alpha=0.05)
            results[cat]['Tukey_HSD'] = tukey.summary()
            print(f"{category}: {cat}, rating column: {rating_column}")
            print(f"ANOVA p-value: {p_value:.4f} - Significant difference between {timescale}")
            print(tukey.summary())
        else:
            print(f"{category}: {cat}, rating column: {rating_column}")
            print(f"ANOVA p-value: {p_value:.4f} - No significant difference between {timescale}")

        print("-" * 50)

    return results
