import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getTheData():
	boston_housing = keras.datasets.boston_housing
	(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

	order = np.argsort(np.random.random(train_labels.shape))
	train_data = train_data[order]
	train_labels = train_labels[order]
	return (train_data, train_labels), (test_data, test_labels)

def seeThePandasDataFrameHead(df):
	head = df.head()
	print(head)

def buildTheModel(train_data):
	model = keras.Sequential([
		keras.layers.Dense(64, activation=tf.nn.relu, 
							input_shape=(train_data.shape[1],)),
		keras.layers.Dense(64, activation=tf.nn.relu),
		keras.layers.Dense(1)
	])

	optimizer = tf.train.RMSPropOptimizer(0.001)

	model.compile(loss='mse',
					optimizer=optimizer,
					metrics=['mae'])
	return model


def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [1000$]')
	plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
			   label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
			   label = 'Val loss')
	plt.legend()
	plt.ylim([0,5])
	plt.show()


def main():
	(train_data, train_labels), (test_data, test_labels) = getTheData()

	column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
				'TAX', 'PTRATIO', 'B', 'LSTAT']

	df = pd.DataFrame(train_data, columns=column_names)
	seeThePandasDataFrameHead(df)

	# std = standard deviation
	mean = train_data.mean(axis=0)
	std = train_data.std(axis=0)
	train_data = (train_data - mean) / std
	test_data = (test_data - mean) / std


	model = buildTheModel(train_data)
	EPOCHS = 500
	history = model.fit(train_data, train_labels, epochs=EPOCHS,
					validation_split=0.2, verbose=0,
					)
	plot_history(history)
	
	test_predictions = model.predict(test_data).flatten()
	print(test_predictions)

if __name__ == '__main__':
	# disable this warning: 2018-08-19 14:07:23.038744: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	main()