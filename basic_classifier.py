import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def getTheFashionDataset():
	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	return (train_images, train_labels), (test_images, test_labels)


def exploreTheData(train_images, train_labels):
	print('tain images shape ', train_images.shape)
	print('length of train labels ', len(train_labels))
	print('train labels ', train_labels)

def showFirstImage(img):
	plt.figure()
	plt.imshow(img)
	plt.colorbar()
	plt.gca().grid(False)
	plt.show()

def showFirstXImages(images):
	plt.figure(figsize=(10,10))
	i = -1
	for img in images:
		i += 1
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid('off')
		plt.imshow(img, cmap=plt.cm.binary)
	plt.show()

def createTheModel():
	# flatten means: 28*28 2D array --> 784 1D array
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28, 28)),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.softmax)
	])
	model.compile(optimizer=tf.train.AdamOptimizer(), 
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	return model

def trianTheModel(model, train_images, train_labels):
	model.fit(train_images, train_labels, epochs=5)


def evaluateTheModel(model, test_images, test_labels):
	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print('Test accuracy:', test_acc)

def predict(model, test_images):
	predictions = model.predict(test_images)
	print('Predictions array: ', predictions[0])
	print('Prediction result: ', np.argmax(predictions[0]))


def checkPredictionAgainstLabel(model, test_images, test_labels, class_names):
	predictions = model.predict(test_images)
	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid('off')
		plt.imshow(test_images[i], cmap=plt.cm.binary)
		predicted_label = np.argmax(predictions[i])
		true_label = test_labels[i]
		if predicted_label == true_label:
		  color = 'green'
		else:
		  color = 'red'
		plt.xlabel("{} ({})".format(class_names[predicted_label], 
									  class_names[true_label]),
									  color=color)
	plt.show()

def main():
	(train_images, train_labels), (test_images, test_labels) = getTheFashionDataset()
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
	
	exploreTheData(train_images, train_labels)
	showFirstImage(train_images[0])

	train_images = train_images / 255.0
	test_images = test_images / 255.0

	showFirstXImages(train_images[:25])

	model = createTheModel()
	trianTheModel(model, train_images, train_labels)
	evaluateTheModel(model, test_images, test_labels)
	predict(model, test_images)
	checkPredictionAgainstLabel(model, test_images, test_labels, class_names)


if __name__ == '__main__':
	main()