# DataBrewers

## Abstract
Beer reviews offer a unique lens into consumer preferences, revealing intriguing patterns influenced by both seasonal trends and long-term shifts in taste. The goal of our project is to analyze potential patterns in ratings and reviews over time, focusing on how beer preferences evolve across seasons and years. This research could provide valuable insights for professionals such as brewers and marketers, enabling them to better align their product offerings with consumer demand. By understanding these seasonal trends, brewers can optimize their strategies to cater to shifting preferences.

## Research questions
- How do beer preferences change across different seasons or over the years?
- Do alcohol content and beer style affect ratings depending on the season?
- Does the location of users—and consequently different climate condition—impact their beer preferences?
- Does the alcohol content or beer style impact ratings differently depending on the time of year? 
- Are specific beers consistently associated with certain seasons or weather conditions in beer reviews?

## Proposed additional datasets
We decided that we won't be using any additional datasets for this project as the data provided in the BeerAdvocate dataset are enough to conduct a seasonal analysis of beer reviews.

## Data 
Our dataset includes two dataset from two different websites : BeerAdvocate and RateBeer. In data preprocessing, we decided to analyze only the reviews published on the BeerAdvocate website, by users based in the United States of America. The reason of this choice is that most of the reviews are done in this country and it provides us with enough data to conduct our analysis. This lead us with more than 100'000 users and 7’000’000 ratings, among which we have more than 2’000’000 reviews. To conduct the season analysis through the year, we focused on the year 2006 to 2017 to have a minimal threshold of 70’000 reviews by year. 

**The datasets used in this project are downloadable through the [following link](https://drive.google.com/drive/folders/1Wz6D2FM25ydFw_-41I9uTwG9uNsN4TCF?usp=share_link)**.

## Methods

## Task 1: Data preprocessing and merging
To ensure efficient data preprocessing for analyzing beer preferences, we focused on loading, cleaning, and merging datasets to create a comprehensive structure for analysis, following these steps:

- **Load raw data files:** Imported datasets, including reviews, ratings, user profiles, and brewery information.
- **Handle missing values and normalize columns:** Addressed missing entries, removed invalid data points, and converted categorical data into usable formats
- **Analyze dataset content:** Evaluated the data to determine if additional datasets were required.
- **Restrict analysis to USA-based BeerAdvocate users:** Verified that limiting the scope to users within the United States in the BeerAdvocate dataset did not pose significant limitations.
- **Categorization:** Group states based on geographical regions, classify beer styles into broader categories, and divide numerical alcohol content into three distinct ranges.
---

## Task 2: Temporal analysis of beer preferences
To uncover seasonal trends in beer preferences, we followed these steps to analyze variations in beer ratings and characteristics over time:

- **Seasonal analysis:** Examined how beer ratings and characteristics change across seasons.
- **Analyze review frequency:**
    - Calculated the total number of reviews submitted each year.
    - Investigated the number of reviews by season to uncover temporal trends in beer popularity.
- **Average score analysis:**
    - Calculated the average score per year based on the alcohol percentage of beers.
    - Determined the average score by year for each beer style to identify trends in style preferences.
- **Temporal trend visualization:** Created visual representations of seasonal and event-driven patterns to highlight key insights.

---

## Task 3: Comparative analysis of beer preferences across states
To uncover regional trends and cultural influences, we analyzed geographic variations in beer preferences by following these steps:

- **Regional trends in ratings:**
    - Compared the average beer ratings across states to uncover geographic patterns in beer appreciation.
    - Examined differences in review frequency by state to identify regions with higher engagement in beer culture.
    - Compared the preferred beer styles by states and by seasons 

---

## Task 4: Sentiment and semantic analysis
To understand subjective aspects of beer preferences and their associations with seasons and events, we examined textual reviews using the following approach:

- **Word Frequency Analysis:** Identified the most frequent words in reviews across all data.
- **Ambiguous Words:**
    - Analyzed ambiguous words to classify them as positive or negative based on associated ratings.
    - Manually tested ambiguous word classifications using 100+ reviews, removing words misclassified more than 20% of the time.
- **Seasonal and Event Associations:**
    - Identified words linked to specific seasons (e.g., "fresh" and "light" for summer).
- **Comparison with Ratings:**
    - Compared word frequency with the six metrics (taste, aroma, palate, overall rating, review count) to find correlations.
    - Analyzed changes in word usage across seasons to highlight evolving trends.


## Proposed timeline

- **Data preprocessing and exploration**: 15th November
- **Task implementation**: 29th November
- **Final analysis**: 15th December
- **Data story**: 20th December

## Organization within the team
- Eugénie Cyrot: Data exploration analysis (ratings), seasonal trend analysis (years, style, US level), plot introduction datastory
- Clémence Kiehl: Data exploration analysis (reviews), sentiment analysis, Semantic analysis by season
- Marin Philippe: Data exploration analysis (reviews, beers style), beer style analysis, semantic analysis,
- Cléo Renaud: Sentiment analysis, modularization sentiment analysis, website for the datastory, READme, text for introduction datastory
- Pauline Theimer-Lienhard: Data exploration analysis (ratings), seasonal trend analysis (state level, alcohol), organization of the repository (scripts, notebook)

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
├── data                            <- Contains all project data files (where all downloaded files should be)
│   ├── cleaned                     <- Stores all processed and cleaned datasets
│
├── src                         
│   ├── graph                       <- Contains all the graph visualizations in HTLM for the data story
│   ├── scripts     
│   │   ├── categorization.py       <- Contains the functions for categorizing the dataset (e.g. by ABV, styles)
│   │   ├── dataloader.py           <- Contains the functions to load the datasets from raw files
│   │   ├── plot.py                 <- Contains the functions to plot
│   │   ├── preprocessing.py        <- Main script for preprocessing and cleaning the dataset
│   │   ├── semantic_analysis.py    <- Contains the functions for semantic anylsis
│   │   ├── sentiment_analysis.py   <- Contains the functions for sentiment anylsis
│   │   ├── statistics.py           <- Contains the functions for statistical testing and analysis
│
├── data_cleaning.ipynb             <- Notebook for exploratory data analysis and data cleaning
├── graph_html.ipynb           <- Notebook showing the graph for the datastory
├── milestoneP2.ipynb               <- Notebook showing the results for the milestone P2
├── milestoneP3.ipynb               <- Notebook showing the results for the milestone P3
├── seasonal_trend_analysis.ipynb   <- Notebook containing the detailled analysis over the seasons and years
│
├── config.yml                      <- For the datastory website
├── Gemfile                         <- For the datastory website
├── index.md                        <- For the datastory website
├── .gitignore                      <- List of files ignored by git
├── pip_requirements.txt            <- File for installing python dependencies
└── README.md
```

