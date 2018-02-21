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


# Plot and save histogram
#   - dscr : title of plot
#   - filename : if save is True, plot will be saved as plots/filename.png
#   - save : if False, plot will pop-up but not be saved
def plot_histogram(data, dscr, filename=None, save=False):
    
    plt.hist(data, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], rwidth=0.8)
    plt.title(dscr)
    plt.xlabel('Rating')
    
    if save:
        plt.savefig(os.path.join('plots', filename + '.png'))
    else:
        plt.show()


if __name__ == "__main__":

    data, movies = load_data()
    
    # plot histogram for all ratings
    plot_histogram(data['rating'], 'All Ratings')

    # Example for saving plot:
    # plot_histogram(data['rating'], 'All Ratings', filename='4_1', save=True)

# TODO: plot histogram for...
# 2. All ratings of the ten most popular movies (movies which have received the most ratings).
# 3. All ratings of the ten best movies (movies with the highest average ratings).
# 4. All ratings of movies from three genres of your choice (create three separate visualizations).