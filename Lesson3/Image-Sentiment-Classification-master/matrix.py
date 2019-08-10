import os
import sys
import model
import matplotlib
matplotlib.use('Agg')
import itertools
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import confusion_matrix
np.random.seed(0)

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
	"""
	This function prints and plots the confusion matrix.
	"""
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def main():
	# set mode
	try:
		mode = sys.argv[1]
		assert (mode == 'dnn' or mode == 'cnn')
	except:
		print('Error: Model mode not found')
		exit()

	# load data	
	_, te_feats, _, te_labels = read_dataset()
	
	# load model
	# TODO load your model here

	# predict
	# TODO use 'predictions = your_model.predict_classes(te_feats)' to get predictions
	
	# one-hot to int
	te_labels = np.argmax(te_labels, axis = -1)
	
	# draw confsion matrix
	conf_mat = confusion_matrix(te_labels, predictions)
	plt.figure()
	plot_confusion_matrix(conf_mat, classes = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
	plt.savefig('confusion_matrix_'+mode+'.png')

if __name__ == '__main__':
	main()
