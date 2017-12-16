'''
Created on Dec 12, 2017

@author: martin
'''

import matplotlib.pyplot as plt

def plot_corr(df, fn=None, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns. Blue-cyan-yellow-red-darkred => less to more correlated
                                               0 ------------------>  1
                                               Expect a darkred line in the diagonal
    """
    
    corr= df.corr()
    fig, ax= plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    if fn != None:
        plt.savefig(fn)
    
    print("saved")
    
    return fig


if __name__ == '__main__':
    pass