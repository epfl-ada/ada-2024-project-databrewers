import pandas as pd
from categorization import assign_region, categorize_abv, get_season, get_style_map
from dataloader import load_beer_advocate_data, format_data

# Load BeerAdvocate data
beers_ba, reviews_ba, users_ba, breweries_ba, ratings_ba = load_beer_advocate_data()

# Reshape reviews and ratings
reviews_ba = format_data(reviews_ba, "review_id")
ratings_ba = format_data(ratings_ba, "rating_id")

# Convert the number of ratings values to numeric
users_ba['nbr_ratings'] = pd.to_numeric(users_ba['nbr_ratings'], errors='coerce')
users_ba['nbr_reviews'] = pd.to_numeric(users_ba['nbr_reviews'], errors='coerce')


## LOCATION

# Merge the reviews and ratings with user information on location
reviews_ba = reviews_ba.merge(users_ba[['user_id', 'location']], on='user_id', how='left')
ratings_ba = ratings_ba.merge(users_ba[['user_id', 'location']], on='user_id', how='left')

# Remove the states to only have the countries name
users_ba['country'] = users_ba['location'].str.split(',').str[0]
reviews_ba['country'] = reviews_ba['location'].str.split(',').str[0]
ratings_ba['country'] = ratings_ba['location'].str.split(',').str[0]

# Remove the countries to only have the states name
reviews_ba['states'] = reviews_ba['location'].str.split(',').str[1]
ratings_ba['states'] = ratings_ba['location'].str.split(',').str[1]
users_ba['states'] = users_ba['location'].str.split(',').str[1]

# Replace missing 'country' values with a label "Unknown"
reviews_ba['country'] = reviews_ba['country'].fillna('Unknown')
ratings_ba['country'] = ratings_ba['country'].fillna('Unknown')
users_ba['country'] = users_ba['country'].fillna('Unknown')

# Keep only users from the US
reviews = reviews_ba[reviews_ba['country'] == 'United States']
ratings = ratings_ba[ratings_ba['country'] == 'United States']
users = users_ba[users_ba['country'] == 'United States']

# Assign region to US state
reviews['region'] = reviews['states'].apply(assign_region)
ratings['region'] = ratings['states'].apply(assign_region)
users['region'] = users['states'].apply(assign_region)


## BEER STYLE

# Remove the country in the beer name
beers_ba['style'] = beers_ba['style'].str.replace('American ', '')
beers_ba['style'] = beers_ba['style'].str.replace('German ', '')
beers_ba['style'] = beers_ba['style'].str.replace('Czech ', '')
beers_ba['style'] = beers_ba['style'].str.replace('Belgian ', '')
beers_ba['style'] = beers_ba['style'].str.replace('English ', '')
beers_ba['style'] = beers_ba['style'].str.replace('Euro ', '')
beers_ba['style'] = beers_ba['style'].str.replace('Scottish ', '')
beers_ba['style'] = beers_ba['style'].str.replace('Double / Imperial ', '')
beers_ba['style'] = beers_ba['style'].str.replace(' (APA)', '')

# New style map
stylemap = get_style_map()
beers_ba['style_simp'] = beers_ba['style'].replace(stylemap)

# Add the beer style simplified
reviews = reviews.merge(beers_ba[['beer_id', 'style_simp']], on='beer_id', how='left')
ratings = ratings.merge(beers_ba[['beer_id', 'style_simp']], on='beer_id', how='left')


## ABV

# Convert 'abv' to numeric, coercing errors to NaN
reviews['abv'] = pd.to_numeric(reviews['abv'], errors='coerce')
ratings['abv'] = pd.to_numeric(ratings['abv'], errors='coerce')

# Calculate the quantiles
abv_quantiles = reviews['abv'].quantile([0.25, 0.5, 0.75])

# Categorize the abv
reviews['abv_category'] = reviews['abv'].apply(lambda x: categorize_abv(x, abv_quantiles))
ratings['abv_category'] = ratings['abv'].apply(lambda x: categorize_abv(x, abv_quantiles))


## SEASON

# Extract the year, month, and day
reviews['date'] = pd.to_datetime(reviews['date'], unit='s', errors='coerce')
reviews['year'] = reviews['date'].dt.year
reviews['month'] = reviews['date'].dt.month
reviews['day'] = reviews['date'].dt.day

ratings['date'] = pd.to_datetime(ratings['date'], unit='s', errors='coerce')
ratings['year'] = ratings['date'].dt.year
ratings['month'] = ratings['date'].dt.month
ratings['day'] = ratings['date'].dt.day

# Separate the reviews and ratings by seasons
reviews['season'] = reviews['month'].apply(get_season)
ratings['season'] = ratings['month'].apply(get_season)

# Order the seasons
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
reviews['season'] = pd.Categorical(reviews['season'], categories=season_order, ordered=True)
ratings['season'] = pd.Categorical(ratings['season'], categories=season_order, ordered=True)

## YEAR

# Filter to keep only years with more than 70'000 reviews
reviews_per_year = reviews['year'].value_counts()
years_above_threshold = reviews_per_year[reviews_per_year > 70000].index

reviews = reviews[reviews['year'].isin(years_above_threshold)]
ratings = ratings[ratings['year'].isin(years_above_threshold)]



# Remove Nan values
ratings = ratings.dropna(subset=['rating'])


## SAVE

reviews.to_csv('data/cleaned/reviews.csv.gz', index=False, compression='gzip')
ratings.to_csv('data/cleaned/ratings.csv.gz', index=False, compression='gzip')
users.to_csv('data/cleaned/users.csv.gz', index=False, compression='gzip')
beers_ba.to_csv('data/cleaned/beers.csv.gz', index=False, compression='gzip')
breweries_ba.to_csv('data/cleaned/breweries.csv.gz', index=False, compression='gzip')