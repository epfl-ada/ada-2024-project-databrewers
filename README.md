# DataBrewers

## Abstract
Beer reviews offer a unique lens into consumer preferences, revealing intriguing patterns influenced by both seasonal trends and long-term shifts in taste. The goal of our project is to analyze potential patterns in ratings and reviews over time, focusing on how beer preferences evolve across seasons and years. This research could provide valuable insights for professionals such as brewers and marketers, enabling them to better align their product offerings with consumer demand. By understanding these seasonal trends, brewers can optimize their strategies to cater to shifting preferences.
As the conclusion of our research, we aim to create a "time fresco" that suggests the ideal beer for each season or festivity, offering a visually engaging guide for both industry professionals and enthusiasts.

## Research questions
- Are there some notable patterns through the seasons or the years ?
- How do beer ratings vary by season ?
- Which beer characteristics are the most important through the seasons ?
- Are there some beers that are more consumed during specific seasons ?
- Are certain words or descriptors more frequently used in beer reviews during specific seasons ?

## Proposed additional datasets
We decided that we won't be using any additional datasets for this project as the data provided in the BeerAdvocate dataset are enough to conduct a seasonal analysis of beer reviews.

## Data 
Our dataset includes two dataset from two different websites : BeerAdvocate and RateBeer. In data preprocessing, we decided to analyze only the reviews published on the BeerAdvocate website, by users based in the United States of America. The reason of this choice is that the vast majority of the reviews are done in this country and it provides us with enough data to conduct our analysis. This lead us with more than 100'000 users and 7’000’000 ratings, among which we have more than 2’000’000 reviews. To conduct the season analysis through the year, we focused on the year 2006 to 2017 to have a minimal threshold of 70’000 reviews by year. 

**The datasets used in this project are downloadable through [following link](https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF?usp=share_link)**.

## Methods
In this project, we used the following methods to process our data:

### Task 1: Data preprocessing and merging
Efficient data processing is crucial for analyzing beer preferences. This task involves loading, cleaning, and merging datasets to create a comprehensive data structure for analysis.

#### Task 1.1: Data loading and merging
- Load raw data files, including reviews, ratings, users, and breweries.
- Merge the datasets based on common identifiers to create a unified dataset for analysis.

#### Task 1.2: Data cleaning and transformation
- Handle missing values, remove invalid data entries.
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

## Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-databrewers.git
cd ada-2024-project-databrewers

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>

# install requirements
pip install -r pip_requirements.txt
```

## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

