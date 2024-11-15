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

stylemap = {
"Bock" : "Bock",
"Doppelbock" :"Bock",
"Eisbock" : "Bock",
"Maibock" : "Bock",
"Weizenbock" : "Bock",
"Altbier":"Brown Ale",
"American Brown Ale":"Brown Ale",
"Belgian Dark Ale":"Brown Ale",
"English Brown Ale":"Brown Ale",
"English Dark Mild Ale":"Brown Ale",
"Dubbel":"Dark Ale",
"Roggenbier":"Dark Ale",
"Scottish Ale ":"Dark Ale",
"Winter Warmer":"Dark Ale",
"American Amber / Red Lager" : "Dark Lager",
"Czech Amber Lager" : "Dark Lager",
"Czech Dark Lager" : "Dark Lager",
"European Dark Lager" : "Dark Lager",
"Märzen" : "Dark Lager",
"Munich Dunkel" : "Dark Lager",
"Rauchbier" : "Dark Lager",
"Schwarzbier" : "Dark Lager",
"Vienna Lager" : "Dark Lager",
"Bière de Champagne / Bière Brut":"Hybrid Beer",
"Braggot":"Hybrid Beer",
"California Common / Steam Beer":"Hybrid Beer",
"Cream Ale":"Hybrid Beer",
"American IPA":"IPA",
"Belgian IPA":"IPA",
"Black IPA":"IPA",
"Brut IPA":"IPA",
"English IPA":"IPA",
"Imperial IPA":"IPA",
"Milkshake IPA":"IPA",
"New England IPA":"IPA",
"American Double / Imperial IPA":"IPA",
"American Amber / Red Ale":"Pale Ale",
"American Blonde Ale":"Pale Ale",
"American Pale Ale":"Pale Ale",
"Belgian Blonde Ale":"Pale Ale",
"Belgian Pale Ale":"Pale Ale",
"Bière de Garde":"Pale Ale",
"English Bitter":"Pale Ale",
"English Pale Ale":"Pale Ale",
"English Pale Mild Ale":"Pale Ale",
"Extra Special / Strong Bitter (ESB)":"Pale Ale",
"Grisette":"Pale Ale",
"Irish Red Ale":"Pale Ale",
"Kölsch":"Pale Ale",
"Saison":"Pale Ale",
"Saison / Farmhouse Ale":"Pale Ale",
"American Adjunct Lager":"Pale Lager",
"American Lager":"Pale Lager",
"Bohemian / Czech Pilsner":"Pale Lager",
"Czech Pale Lager":"Pale Lager",
"European / Dortmunder Export Lager":"Pale Lager",
"European Pale Lager":"Pale Lager",
"European Strong Lager":"Pale Lager",
"Festbier / Wiesnbier":"Pale Lager",
"German Pilsner":"Pale Lager",
"Helles":"Pale Lager",
"Imperial Pilsner":"Pale Lager",
"India Pale Lager (IPL)":"Pale Lager",
"Kellerbier / Zwickelbier":"Pale Lager",
"Light Lager":"Pale Lager",
"Malt Liquor":"Pale Lager",
"American Porter": "Porter",
"Baltic Porter": "Porter",
"English Porter": "Porter",
"Imperial Porter": "Porter",
"Robust Porter": "Porter",
"Smoked Porter": "Porter",
"Chile Beer":"Speciality Beer",
"Fruit and Field Beer":"Speciality Beer",
"Gruit / Ancient Herbed Ale":"Speciality Beer",
"Happoshu":"Speciality Beer",
"Herb and Spice Beer":"Speciality Beer",
"Japanese Rice Lager":"Speciality Beer",
"Kvass":"Speciality Beer",
"Low-Alcohol Beer":"Speciality Beer",
"Pumpkin Beer":"Speciality Beer",
"Rye Beer":"Speciality Beer",
"Sahti":"Speciality Beer",
"Smoked Beer":"Speciality Beer",
"American Imperial Stout":"Stout",
"American Stout":"Stout",
"English Stout":"Stout",
"Foreign / Export Stout":"Stout",
"Irish Dry Stout":"Stout",
"Oatmeal Stout":"Stout",
"Russian Imperial Stout":"Stout",
"Sweet / Milk Stout":"Stout",
"American Barleywine":"Strong Ale",
"American Strong Ale":"Strong Ale",
"Belgian Dark Strong Ale":"Strong Ale",
"Belgian Pale Strong Ale":"Strong Ale",
"English Barleywine":"Strong Ale",
"English Strong Ale":"Strong Ale",
"Imperial Red Ale":"Strong Ale",
"Old Ale":"Strong Ale",
"Quadrupel (Quad)":"Strong Ale",
"Scotch Ale / Wee Heavy":"Strong Ale",
"Tripel":"Strong Ale",
"Wheatwine":"Strong Ale",
"American Dark Wheat Beer":"Wheat Beer",
"American Pale Wheat Beer":"Wheat Beer",
"Dunkelweizen":"Wheat Beer",
"Grodziskie":"Wheat Beer",
"Hefeweizen":"Wheat Beer",
"Kristallweizen":"Wheat Beer",
"Witbier":"Wheat Beer",
"Berliner Weisse": "Wild Beer",
"Brett Beer": "Wild Beer",
"Faro": "Wild Beer",
"Flanders Oud Bruin": "Wild Beer",
"Flanders Red Ale": "Wild Beer",
"Fruit Lambic": "Wild Beer",
"Fruited Kettle Sour": "Wild Beer",
"Gose": "Wild Beer",
" Wild Beer": "Wild Beer",
"Gueuze": "Wild Beer",
"Lambic": "Wild Beer",
"Wild Ale": "Wild Beer",
"India Pale Ale (IPA)": "IPA",
"Maibock / Helles Bock": "Bock", 
}

def get_style_map():
    return stylemap