import pandas as pd
import numpy as np

import sys
from os import listdir
from os.path import isfile, join

from sklearn import preprocessing
from keras.models import load_model
from datagenerator import KerasDataGenerator

def create_segments(test_path):
    segment_ids = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    X = []
    for seg_id in segment_ids:
        file = test_path + seg_id
        testdata = pd.read_csv(file).values
        testobject = KerasDataGenerator(testdata)
        test_X = testobject.create_X()
        X.append(test_X)
    return X, segment_ids

def main(modelname):
    modelfile = '../models/' + modelname + '.hdf5'
    model = load_model(modelfile)
    test_segs, segment_ids = create_segments('../test/')
    predictions = pd.DataFrame(columns=['seg_id','time_to_failure'])

    for index, test in enumerate(test_segs):
        time = model.predict(np.expand_dims(test,0))[0][0]
        df = pd.DataFrame([[segment_ids[index][:-4], time]], columns=['seg_id','time_to_failure'])
        predictions = predictions.append(df, ignore_index=True)

    predictions.to_csv('../submissions/submission05.csv', index=False)

if __name__ == '__main__':
    modelname = sys.argv[1]
    main(modelname)
