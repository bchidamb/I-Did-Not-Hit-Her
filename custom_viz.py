from process_data import *
import prob2utils_withbias as model_2
from matrixfact_viz import project_to_2D
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import os


genres=['action', 'childrens', 'horror', 'romance', 'sci-fi']


# Plot ten movies using their 2D representations 
#   - movies_to_plot : list of movie_ids for the ten movies to be plotted
#       NOTE: movie_ids are between 1 and 1682, inclusive
#   - dscr : title of plot
#   - filename : if save is True, plot will be saved as plots/filename.png
#   - save : if False, plot will pop-up but not be saved
def custom_visualize(V_proj, g_ids, dscr, filename=None, save=False):

    plt.figure()
    
    # normalize data
    movie_arr = np.array(V_proj)
    x_mean, x_std = np.mean(movie_arr[:, 0]), np.std(movie_arr[:, 0])
    y_mean, y_std = np.mean(movie_arr[:, 1]), np.std(movie_arr[:, 1])
    movie_arr[:, 0] = (movie_arr[:, 0] - x_mean) / x_std
    movie_arr[:, 1] = (movie_arr[:, 1] - y_mean) / y_std
    
    # plot transformed data
    for i, genre in enumerate(genres):
    
        plt.plot(0, 0, '.', color='C'+str(i), label=genre)
    
        for m in g_ids[i]:
            
            x, y = movie_arr[m-1]
            plt.plot(x, y, '.', color='C'+str(i))
        
    plt.title(dscr)
    plt.legend()
    
    if save:
        plt.savefig(os.path.join('plots', filename + '.png'))
    else:
        plt.show()
        

# Collect movie_ids per genre
g_ids = [[] for genre in genres]
        
for i, genre in enumerate(genres):
    g_ids[i] = movies.loc[movies[genre] == 1, 'movie_id'].values


Y_train = np.loadtxt('data/train.txt').astype(int)
Y_test = np.loadtxt('data/test.txt').astype(int)

M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies

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
custom_visualize(V_proj_2, g_ids, 'Model 2: Movies by Genre')
    