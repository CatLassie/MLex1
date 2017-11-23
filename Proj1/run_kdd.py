'''
Created on Nov 22, 2017

@author: martin
'''

import pandas as pd
from TGaussianNB import TGaussianNB

def main():
    h= open("../data/kddcup.names").read().split(".\n")
    labels= [l+'.' for l in h[0].split(",")]
    cols= [x.split(":")[0] for x in h[1:len(h)-1]] + ['class']
    
    # data= pd.read_csv("../data/kddcup.data_10_percent_corrected", names=cols)
    data= pd.read_csv("../data/kddcup.data.corrected", names=cols)
    
    # ['duration','protocol_type','service','src_bytes','dst_bytes','flag','land','wrong_fragment','urgent','hot','logged_in','num_compromised','root_shell',
    # 'su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_hot_login','is_guest_login','count','serror_rate',
    # 'rerror_rate','same_srv_rate','diff_srv_rate','srv_count','srv_serror_rate','srv_rerror_rate','srv_diff_host_rate']
    
    # Features to be used in classification
    features= ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root']
    
    X= data[features]
    y= data['class'].apply(lambda x: labels.index(x))
    
    h= TGaussianNB(X, y)
    h.run()

if __name__ == '__main__':
    main()
