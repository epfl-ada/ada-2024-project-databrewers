import tarfile
import pandas as pd

def load_matched_beer_data():
    # Extract and load 'matched_beer_data.tar.gz'
    file_path = '../../data/matched_beer_data.tar.gz'
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path='../data')

    ratings_merged = pd.read_csv('../data/ratings.csv', low_memory=False)
    users_approx = pd.read_csv('../data/users_approx.csv')
    users_merged = pd.read_csv('../data/users.csv')
    beers_merged = pd.read_csv('../data/beers.csv', low_memory=False)
    breweries_merged = pd.read_csv('../data/breweries.csv')

    return ratings_merged, users_approx, users_merged, beers_merged, breweries_merged

def load_beer_advocate_data():
    # Extract and load 'BeerAdvocate.tar.gz'
    file_path = '../../data/BeerAdvocate.tar.gz'
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path='../data')

    beers_ba = pd.read_csv('../data/beers.csv')
    reviews_ba = pd.read_csv('../data/reviews.txt.gz', header=None, names=["info"], delimiter='\t', on_bad_lines='skip')
    users_ba = pd.read_csv('../data/users.csv')
    breweries_ba = pd.read_csv('../data/breweries.csv')
    ratings_ba = pd.read_csv('../data/ratings.txt.gz', header=None, names=["info"], delimiter='\t', on_bad_lines='skip')

    return beers_ba, reviews_ba, users_ba, breweries_ba, ratings_ba

def load_rate_beer_data():
    # Extract and load 'RateBeer.tar.gz'
    file_path = '../../data/RateBeer.tar.gz'
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path='../data')

    beers_rb = pd.read_csv('../data/beers.csv')
    reviews_rb = pd.read_csv('../data/reviews.txt.gz', header=None, names=["info"], delimiter='\t', on_bad_lines='skip')
    users_rb = pd.read_csv('../data/users.csv')
    breweries_rb = pd.read_csv('../data/breweries.csv')
    ratings_rb = pd.read_csv('../data/ratings.txt.gz', header=None, names=["info"], delimiter='\t', on_bad_lines='skip')

    return beers_rb, reviews_rb, users_rb, breweries_rb, ratings_rb


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


