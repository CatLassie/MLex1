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
    # plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    if fn != None:
        plt.savefig(fn)
    
    print("saved")
    
    return fig

def print_types(df):
    for col in df.columns:
        print(col, df[col].dtype)
        

def writeResultsTable(results, fn):
    scale= 1000
    tt= " (ms)"
    
    out= open(fn,'w')
    out.write("\\begin{tabular}{lrrrrr}\n")
    out.write("\\toprule\n")
    out.write("\\textbf{Algorithm/Setting}")
    out.write(" & \\textbf{time" + tt + "} & \\textbf{time p" + tt + "} & \\textbf{rmse} & \\textbf{mae} & \\textbf{cv}\\\\\n")
    out.write("\\midrule\n")
    
    for alg, res in results.items():
        pc= alg
        
        s= "{6} & ${0:.2f} \\pm {1:.2f}$ & ${2:.2f} \\pm {3:.2f}$ & ${4:.2f} \\pm {5:.2f}$ & ${7:.2f} \\pm {8:.2f}$ & ${9:.2f} \\pm {10:.2f}$\\\\".format(res.tfm*scale,res.tfs*scale, res.tpm*scale,res.tps*scale, res.sm, res.ss, pc, res.mm, res.ms, res.cm, res.cs)
        out.write(s + "\n")
        
    out.write("\\bottomrule\n")
    out.write("\\end{tabular}\n")
    
    

if __name__ == '__main__':
    pass