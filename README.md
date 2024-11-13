# DataBrewers

## Abstract
The aim of our project is to analyze beer preference shifts across seasons and in response to weather changes, events and cultural festivities. This could significantly help professionals (brewers, marketers, etc.) by providing insights into customer preferences. By understanding seasonal trends, brewers can adjust their product offerings to align more closely with consumer demand. 
Moreover, as a conclusion of our research, we could suggest the best beer that would be the perfect match for each season/festivity, in the form of a time fresco. 
(need more words -> 150)

## Research questions
- Are there some notable patterns through the seasons ?
- How do beer ratings vary by season ?
- Which beer characteristics are the most important through the seasons ?
- Are there some beers that are more consumed during specific seasons ?
- How does the weather influence the ratings ?

## Proposed additional datasets
- Potentially: Dataset with the weather in different states through the year. It would enable us to analyze the impact of the weather on the beer choices and the ratings.
  [Weather dataset in the us](https://www.kaggle.com/datasets/nachiketkamod/weather-dataset-us)

## Data 
Our dataset includes two dataset from two different websites : BeerAdvocate and RateBeer. In data preprocessing, we decided to analyze only the reviews published by users based on the United States of America on the BeerAdvocate website. The reason of this choice is that it is the country where the majority of the reviews are done and it provides us with enough data to conduct our analysis. This lead us with … users and more than 7’000’000 ratings, among them more than 2’000’000 reviews. To conduct the season analysis through the year, we focused on the year 2006 to 2017 to have a minimal threshold of 70’000 reviews by year. 

## Methods
In this project, we used the following methods to process our data:

### Task 1: Data preprocessing and merging
Efficient data processing is crucial for analyzing beer preferences. This task involves loading, cleaning, and merging datasets to create a comprehensive data structure for analysis.

#### Task 1.1: Data loading and merging
- Load raw data files, including reviews, ratings, users, and breweries.
- Merge the datasets based on common identifiers to create a unified dataset for analysis.

#### Task 1.2: Data cleaning and transformation
- Handle missing values, remove or impute invalid data entries.
- Normalize columns and standardize formats (e.g., extracting features from text reviews, converting categorical data into usable formats).
- Process user and brewery information (e.g., splitting location into `country` and `state`).

---

### Task 2: Temporal analysis of beer preferences
This task analyzes how beer preferences change over time, focusing on seasonality and festivities.

#### Task 2.1: Seasonal rrends in beer preferences
- Explore how beer ratings and characteristics vary by season (spring, summer, fall, winter).
- Identify beer styles and ratings that are most popular in different seasons.

#### Task 2.2: Effect of festivities on beer preferences
- Investigate how cultural events (e.g., Oktoberfest, Christmas) influence beer ratings and preferences.
- Look for changes in beer consumption patterns during specific events or holidays.

---

### Task 3: Weather-related shifts in beer preferences
This task examines the impact of weather on beer preferences.

#### Weather-related analysis
- Correlate weather variables (temperature, humidity, etc.) with beer ratings.
- Analyze shifts in beer preferences during extreme weather conditions (e.g., hot summers, cold winters).

---

### Task 4: Comparative analysis of beer breferences across states
This task compares beer preferences based on geographic location.

#### State-sased Beer preference analysis
- Investigate how beer preferences differ across states, looking for regional trends.
- Identify beer styles or characteristics that are more popular in specific states.

---

### Task 5: Develop a recommendation model based on season and events
This task aims to create a system that recommends the best beer for each season and event.

#### Recommendation model vevelopment
- Build a recommendation model based on the time of year, weather, and events.
- Use machine learning techniques (e.g., collaborative filtering, content-based filtering) to suggest optimal beer styles for different seasons and festivities.


## Proposed timeline

**Data preprocessing and initial exploratory data analysis**: 15th November
**Task implementation**: 29th November
**Final analysis**: 9th December
**Report**: 15th December
**Data story**: 20th December


## Organization within the team 

| Assignee   | Task |    |
| ---------- |------|----|
| Eugénie    | 2.1  | 
| Clémence   | 2.2  |
| Marin      | 3    | 
| Cléo       | 4    |
| Pauline    | 5    | 


