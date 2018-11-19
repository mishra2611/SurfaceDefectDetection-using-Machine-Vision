import keras
import numpy as np

class Histories(keras.callbacks.Callback):
	def __init__(self, validation_generator, training_generator):
		self.validation_generator = validation_generator
		self.training_generator = training_generator
	
	def on_train_begin(self, logs={}):
		self.aucs = []
		self.binaccuracy = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		for x_train, y_train in self.training_generator.__getitem__:
		#y_pred = self.model.predict(self.validation_data[0])
		#self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
			y_pred_train = np.array(self.model.predict(x_train))
			y_pred_train = y_pred_train.flatten()
			y_train = y_train.flatten()
			countCorrect = 0
			for x in range(len(y_pred_train)):
				if np.sum(y_pred_train[x]) > 10 and np.sum(y_train[x]) > 0 : countCorrect = countCorrect+1
				elif np.sum(y_pred_train[x]) < 10 and np.sum(y_train[x]) == 0 : countCorrect = countCorrect+1
			accu = countCorrect / len(y_pred_train)
			self.binaccuracy.append(accu)
			print("Accuracy: "+str(accu))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
