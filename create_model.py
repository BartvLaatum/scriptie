import numpy as np
import matplotlib.pyplot as plt

from datagenerator import KerasDataGenerator
import load_data
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, CuDNNGRU, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import adam

def convertdata(float_data, step_size):
    X = float_data[:,0]
    y = float_data[:,1]
    rows = X.shape[0]//1000
    X = X[:rows*1000]
    X = X.reshape(rows,1000)
    y = [y[t] for t in range(999,len(y),1000)]
    return X, np.array(y)

def validation_split(data, ratio):
    slice_index = int(len(data) * ratio)
    train_data = data[:slice_index]
    valid_set = data[slice_index:]
    return train_data, valid_set

def built_model(train_gen, valid_gen, hidden_size):
	model = Sequential()
	model.add(CuDNNGRU(hidden_size,input_shape=(None, train_gen.n_features)))
	# model.add(BatchNormalization())
	model.add(Dropout(.2))
	# model.add(CuDNNGRU(hidden_size, return_sequences=True))
	# model.add(BatchNormalization())
	# model.add(Dropout(.2))
	# model.add(Flatten())
	model.add(Dense(10, activation='relu'))
	# model.add(BatchNormalization())
	model.add(Dropout(.2))
	model.add(Dense(1))
	model.compile(optimizer=adam(lr=0.0005), loss="mae")
	print(model.summary())
	return model

def plot_loss(history):
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

def main():
	quakeind1 = 5656574
	quakeind2 = 104677355
	
	print('Loading training file')
	train_df = load_data.load_trainset('../train/train.csv')
	print('Converting to numpy array')
	float_data = train_df.values
	del train_df
	kf = KFold(n_splits=5)
	for train_index, test_index in kf.split(float_data):
    # i+=1
    #     print("TRAIN:", train_index, "TEST:", test_index)
		train_set, valid_set = float_data[train_index], float_data[test_index]
		del float_data
	# train_set, valid_set = validation_split(float_data, 0.8)
		train_gen = KerasDataGenerator(train_set, n_steps=150, step_size=1000, batch_size=32)
		valid_gen = KerasDataGenerator(valid_set, n_steps=150, step_size=1000, batch_size=32)

		model, cb = built_model(train_gen, valid_gen, hidden_size=48)
		history = model.fit_generator(train_gen.generator(),
												steps_per_epoch=1000,
												epochs=50,
												verbose=2,
												callbacks=cb,
												validation_data=valid_gen.generator(),
												validation_steps=200)

		plot_loss(history)
		# model.save('../models/sixth_model.h5')

if __name__ == '__main__':
	main()
