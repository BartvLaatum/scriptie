import numpy as np
import load_data



def convertdata(float_data, step_size):
    X = float_data[:,0]
    y = float_data[:,1]
    print(y[99019999])
    rows = X.shape[0]//1000
    X = X[:rows*1000]
    X = X.reshape(rows,1000)
    y = [y[t] for t in range(999,len(y),1000)]
    print(y[-1])
    return X, np.array(y)



train_df = load_data.load_trainset('../train/train.csv')
float_data = train_df[:]
