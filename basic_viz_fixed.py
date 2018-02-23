import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import os
from process_data import *

# Plot and save histogram
#   - dscr : title of plot
#   - filename : if save is True, plot will be saved as plots/filename.png
#   - save : if False, plot will pop-up but not be saved
def plot_histogram(d, dscr, names=None, filename=None, save=False):

    plt.figure()

    plt.hist(d, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], rwidth=0.8, label=names)
    plt.title(dscr)
    plt.xlabel('Rating')
    
    if names:
        plt.legend()

    if save:
        plt.savefig(os.path.join('plots', filename + '.png'))
    else:
        plt.show()


if __name__ == "__main__":

    # Plot all ratings
    plot_histogram(data['rating'], 'All Ratings', filename='4_1', save=False)

    # Plot most popular movies
    plot_histogram(total_pop_ratings, 'Most Popular', names=popular_movie_names, filename='4_2',save=False)

    # Plot highest rated movies
    plot_histogram(total_high_ratings, 'Highest Rated', names=highest_rated_movie_names, filename='4_3',save=False)

    # Plot all ratings of movies from three genres (action, horror, childrens)
    # Now sort ratings by the three genres chosen
    for i, genre in enumerate(genres):
        plot_histogram(total_g_ratings[i], genre + ' ratings', filename='4_4_' + genre,save=False)
