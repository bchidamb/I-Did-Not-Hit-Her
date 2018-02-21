import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import os


# Loads data (both ratings and movies) into pandas dataframes
def load_data():

    data = pd.read_csv(
        os.path.join('data', 'data.txt'),
        sep='\t',
        names=['user_id', 'movie_id', 'rating']
    )

    movies = pd.read_csv(
        os.path.join('data', 'movies.txt'),
        sep='\t',
        names=['movie_id', 'movie_title', 'unknown', 'action', 'adventure',
        'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
        'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance',
        'sci-fi', 'thriller', 'war', 'western'],
        encoding='latin1' # fixes bug in reading movies.txt
    )

    print('Data entries:', len(data))
    print('Movie entries:', len(movies))

    return data, movies

# Takes a Pandas dataframe
# fetches and outputs all ratings (list) of a given movie id
def fetch_all_ratings(data, movie_id):
    movie_ratings = data[data['movie_id'] == movie_id]
    rs = movie_ratings['rating'].values
    return rs


# Plot and save histogram
#   - dscr : title of plot
#   - filename : if save is True, plot will be saved as plots/filename.png
#   - save : if False, plot will pop-up but not be saved
def plot_histogram(d, dscr, filename=None, save=False):

    plt.hist(d, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], rwidth=0.8)
    plt.title(dscr)
    plt.xlabel('Rating')

    if save:
        plt.savefig(os.path.join('plots', filename + '.png'))
    else:
        plt.show()


if __name__ == "__main__":

    data, movies = load_data()

    # Create new columns in our movies DataFrame
    # to hold the number of ratings and average rating
    # For a given movie
    movies['num_ratings'] = 0
    movies['avg_rating'] = 0

    # plot histogram for all ratings
    plot_histogram(data['rating'], 'All Ratings')

    # build data for most popular and highest rating
    # for i in range(1, 1682+1):
    #     # Get all movie ratings for a specific movies
    #     movie_ratings = fetch_all_ratings(data, i)
    #
    #     # Compute the number of ratings and average ratings
    #     ri_avg = np.mean(ri)
    #     ri_num = len(ri)
    #
    #     # Add these computed values to our DataFrame
    #     movies.loc[movies['movie_id'] == i, 'num_ratings']= ri_num
    #     movies.loc[movies['movie_id'] == i, 'avg_rating']= ri_avg
    #
    #
    # # Plot most popular (highest number) ratings
    # movies.sort_values('num_ratings', inplace=True, ascending=False)
    # popular_movie_ids = movies[:10]['movie_id'].values
    # total_pop_ratings = []
    # for pop_id in popular_movie_ids:
    #     pop_ratings = data.loc[data['movie_id'] == pop_id, 'rating']
    #     total_pop_ratings.append(pop_ratings)
    #
    # total_pop_ratings = pd.Series(np.array(total_pop_ratings))
    #
    # plot_histogram(total_pop_ratings, 'Most Popular', filename='4_2',save=True)
    #
    #
    # # Plot highest rated movies
    # movies.sort_values('avg_rating', inplace=True, ascending=False)
    # highest_rated_movie_ids = movies[:10]['movie_id'].values
    # total_high_ratings = []
    # for high_id in highest_rated_movie_ids:
    #     high_ratings = data.loc[data['movie_id'] == high_id, 'rating']
    #     total_high_ratings.append(high_ratings)
    #
    # total_high_ratings = pd.Series(np.array(total_high_ratings))
    #
    # plot_histogram(total_high_ratings, 'Highest Rated', filename='4_3',save=True)

    # Example for saving plot:
    # plot_histogram(data['rating'], 'All Ratings', filename='4_1', save=True)

# TODO: plot histogram for...
# 2. All ratings of the ten most popular movies (movies which have received the most ratings).
# 3. All ratings of the ten best movies (movies with the highest average ratings).
# 4. All ratings of movies from three genres of your choice (create three separate visualizations).
