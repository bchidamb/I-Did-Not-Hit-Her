import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import os
import prob2utils as model_1
import prob2utils_withbias as model_2
import surpriseimpl1 as model_3
from basic_viz import load_data

# Tasks:
#  - implement the 3 models 
#    (a) baseline from set 5 [DONE]
#    (b) baseline with bias terms [DONE]
#    (c) off-the-shelf
#  - optimize parameters for first 2 models, with k = 20 fixed
#  - compare each models performance on the test set
#  - visualize results from each model by projecting to 2D
#    (a) Any ten movies of your choice from the MovieLens dataset.
#    (b) The ten most popular movies (movies which have received the most ratings).
#    (c) The ten best movies (movies with the highest average ratings).
#    (d) Ten movies from the three genres you selected in Section 4, Basic Visualizations (for a total of 30
#        movies). Create one visualization, containing ten movies, for each of the three genres you select


# Project U and V matrices to 2D as described in the project description
#   - U : an M x K matrix
#   - V : an N x K matrix
def project_to_2D(U, V):

    A, sigma, B = np.linalg.svd(V.T)
    transform = np.matrix(A[:, :2])
    
    U_proj = U * transform
    V_proj = V * transform
    
    # U_proj is M x 2, V_proj is N x 2
    return U_proj, V_proj


# Plot ten movies using their 2D representations 
#   - movies_to_plot : list of movie_ids for the ten movies to be plotted
#       NOTE: movie_ids are between 1 and 1682, inclusive
#   - dscr : title of plot
#   - filename : if save is True, plot will be saved as plots/filename.png
#   - save : if False, plot will pop-up but not be saved
_, movies = load_data()
def visualize(V_proj, movies_to_plot, dscr, filename=None, save=False):
    
    # normalize data
    movie_arr = np.array(V_proj)
    x_mean, x_std = np.mean(movie_arr[:, 0]), np.std(movie_arr[:, 0])
    y_mean, y_std = np.mean(movie_arr[:, 1]), np.std(movie_arr[:, 1])
    movie_arr[:, 0] = (movie_arr[:, 0] - x_mean) / x_std
    movie_arr[:, 1] = (movie_arr[:, 1] - y_mean) / y_std
    
    # plot transformed data
    for m in movies_to_plot:
        
        x, y = movie_arr[m-1]
        movie_name = movies[movies.movie_id == m].iloc[0]['movie_title']
        plt.plot(x, y, 'o')
        plt.annotate(movie_name, (x, y - 0.1), fontsize=10, ha='center')
        
    plt.title(dscr)
    
    if save:
        plt.savefig(os.path.join('plots', filename + '.png'))
    else:
        plt.show(block=False)
    

if __name__ == "__main__":

    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)
    
    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    
    # Train model 1: baseline model from set 5
    k = 20
    # TODO: optimize these parameters, as well as 'eps'
    reg = 0.1
    eta = 0.03 # learning rate
    # eps = ?
    
    print("Training model 1 with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, k, eta, reg))
    U_1, V_1, e_in_1 = model_1.train_model(M, N, k, eta, reg, Y_train)
    e_out_1 = model_1.get_err(U_1, V_1, Y_test)
    print("model 1 results: e_in = %.3f, e_out = %.3f" % (e_in_1, e_out_1))
    
    # Transform model 1 to 2D
    U_proj_1, V_proj_1 = project_to_2D(U_1, V_1)
    
    # Plot model 1
    visualize(
        V_proj_1, 
        [96, 135, 182, 195, 250, 318, 237, 1095, 257, 288], # mess around with this
        'Model 1: Ten Selected Movies'
    )
    
    # Train model 2: baseline model with bias terms
    k = 20
    # TODO: optimize these parameters, as well as 'eps'
    reg = 0.1
    eta = 0.03 # learning rate
   # eps = ?
    
    print("Training model 2 with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, k, eta, reg))
    U_2, V_2, a_2, b_2, e_in_2 = model_2.train_model(M, N, k, eta, reg, Y_train)
    e_out_2 = model_2.get_err(U_2, V_2, a_2, b_2, Y_test)
    print("model 2 results: e_in = %.3f, e_out = %.3f" % (e_in_2, e_out_2))
    
    # Transform model 2 to 2D
    U_proj_2, V_proj_2 = project_to_2D(U_2, V_2)
    
    # Plot model 2
    visualize(
        V_proj_2, 
        [96, 135, 182, 195, 250, 318, 237, 1095, 257, 288], # mess around with this
        'Model 2: Ten Selected Movies'
    )
    
    # TODO: get error out
    print("Training model 3 with M = %s, N = %s, k = %s, eta = %s, reg = %s"%(M, N, k, eta, reg))

    # Note that surprise needs to be passed the data paths since it wants to use the data in
    # a certain way. Can update later to pass in the raw pandas dataframes, this works though
    U_3, V_3, a_3, b_3, e_in_3, e_out_3 = model_3.train_model('./data/train.txt', '.data/test.txt', k, eta, reg, Y_train)

    #e_out_3 = -1
    print("model 3 results: e_in = %.3f, e_out = %.3f" % (e_in_3, e_out_3))
    
    # Transform model 3 to 2D
    U_proj_3, V_proj_3 = project_to_2D(U_3, V_3)
    
    # Plot model 3
    visualize(
        V_proj_3, 
        [96, 135, 182, 195, 250, 318, 237, 1095, 257, 288], # mess around with this
        'Model 3: Ten Selected Movies'
    )

    # add needing an input at the end so program doesn't close so fast
    input('please end me')

    








