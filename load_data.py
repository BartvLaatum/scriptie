import pandas as pd
import numpy as np

def load_trainset(path):
	return pd.read_csv(path, dtype={'acoustic_data': np.int8, 'time_to_failure': np.float64})



# load_trainset('../train/train.csv')