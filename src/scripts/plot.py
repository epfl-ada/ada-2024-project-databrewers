import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

def plot_highest_rated_styles_by_season(highest_rated_style):
    """
       Plot the highest-rated beer styles for each season by US state

       Parameters:
       - highest_rated_style (pd.DataFrame): DataFrame containing the highest-rated beer styles for each state and season
       """

    # Load a us map
    us_states = gpd.read_file \
        ('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')

    # Merge with the highest-rated style data using state names
    merged = us_states.merge(highest_rated_style, left_on='name', right_on='states', how='left')

    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    styles = highest_rated_style['style_simp'].unique()

    # Explicit color map
    style_colors = dict(zip(styles, plt.cm.tab20.colors[:len(styles)]))

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()

    for i, season in enumerate(seasons):
        season_data = merged[merged['season'] == season].copy()  # Explicitly create a copy

        # Map the style_simp column to the explicit colors
        season_data.loc[:, 'color'] = season_data['style_simp'].map(style_colors)

        season_data.plot(
            color=season_data['color'],
            linewidth=0.8,
            ax=axes[i],
            legend=True
        )
        axes[i].set_title(f'Highest rated beer styles for {season}', fontsize=14)
        axes[i].axis('off')

    legend_elements = [
        mlines.Line2D([0], [0], color=color, marker='o', linestyle='None', markersize=10, label=style)
        for style, color in style_colors.items() if style in merged['style_simp'].unique()
    ]
    plt.legend(handles=legend_elements, loc='upper left', title="Beer styles", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_high_low_abv_trends(high_abv, low_abv):
    """"
    Plot mean ratings for high and low ABV beers across seasons

    Parameters:
    - high_abv (pd.DataFrame): DataFrame containing mean ratings for high ABV beers grouped by season
    - low_abv (pd.DataFrame): DataFrame containing mean ratings for low ABV beers grouped by season

    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Low ABV
    axes[0].plot(low_abv['season'], low_abv['rating'])
    axes[0].set_title('Low ABV ratings by season')
    axes[0].set_ylabel('Mean rating')
    axes[0].set_xlabel('Season')
    axes[0].grid(True)

    # Adjust Y-axis limits
    axes[0].set_ylim(low_abv['rating'].min() - 0.1, low_abv['rating'].max() + 0.1)

    # High ABV
    axes[1].plot(high_abv['season'], high_abv['rating'])
    axes[1].set_title('High ABV ratings by season')
    axes[1].set_ylabel('Mean rating')
    axes[1].set_xlabel('Season')
    axes[1].grid(True)

    # Adjust Y-axis limits
    axes[1].set_ylim(high_abv['rating'].min() - 0.1, high_abv['rating'].max() + 0.1)

    plt.tight_layout()
    plt.show()


def plot_abv_seasonal_trends(data, metric):
    """
    Plots seasonal trends of beer ratings for low and high ABV categories across regions

    Parameters:
    - data: The dataset containing the ratings
    - metric: The column name representing the metric to plot
    """

    grouped_data = data.groupby(['season', 'region', 'abv_category'], observed=False)[metric].mean().reset_index()

    plt.figure(figsize=(10, 5))

    # Low ABV
    plt.subplot(1, 2, 1)
    low_abv_data = grouped_data[grouped_data['abv_category'] == 'low']
    sns.lineplot(data=low_abv_data, x='season', y=metric, hue='region', style='region', markers=True, dashes=False)
    plt.xlabel('Season')
    plt.ylabel(metric.capitalize())
    plt.title("Low ABV beers")
    plt.xticks(rotation=45)
    plt.legend(title='Region', loc='upper left', bbox_to_anchor=(1, 1))

    # High ABV
    plt.subplot(1, 2, 2)
    high_abv_data = grouped_data[grouped_data['abv_category'] == 'high']
    sns.lineplot(data=high_abv_data, x='season', y=metric, hue='region', style='region', markers=True, dashes=False)
    plt.xlabel('Season')
    plt.ylabel(metric.capitalize())
    plt.title("High ABV beers")
    plt.xticks(rotation=45)
    plt.legend(title='Region', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def plot(data, x, y, hue, legend_title):
    '''
    Plot the data with x, y, hue and legend tittle specified
    '''
    plt.figure(figsize=(7, 3))

    sns.lineplot(data=data, x=x, y=y, hue=hue, markers=True, dashes=False)

    plt.xlabel(x)
    plt.ylabel(f'{y.capitalize()}')
    plt.xticks(rotation=45)

    plt.legend(title=legend_title, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9, title_fontsize=12)


    plt.tight_layout()
    plt.show()