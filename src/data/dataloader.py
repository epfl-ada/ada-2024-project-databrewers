import tarfile
import pandas as pd

def load_matched_beer_data():
    # Extract and load 'matched_beer_data.tar.gz'
    file_path = 'data/matched_beer_data.tar.gz'
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
    file_path = 'data/BeerAdvocate.tar.gz'
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
    file_path = 'data/RateBeer.tar.gz'
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path='../data')

    beers_rb = pd.read_csv('../data/beers.csv')
    reviews_rb = pd.read_csv('../data/reviews.txt.gz', header=None, names=["info"], delimiter='\t', on_bad_lines='skip')
    users_rb = pd.read_csv('../data/users.csv')
    breweries_rb = pd.read_csv('../data/breweries.csv')
    ratings_rb = pd.read_csv('../data/ratings.txt.gz', header=None, names=["info"], delimiter='\t', on_bad_lines='skip')

    return beers_rb, reviews_rb, users_rb, breweries_rb, ratings_rb




