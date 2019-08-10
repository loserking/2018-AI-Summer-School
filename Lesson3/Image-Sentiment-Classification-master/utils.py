import os
import numpy as np
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
np.random.seed(0)

def read_dataset(tpath = 'data.csv', norm = True, balance = True):
	datas = []
	with open(tpath) as file:
		for line in file.readlines()[1:]:
			line = line.split('\n')[0]
			label, feat = line.split(',')
			feat = np.fromstring(feat, dtype = int, sep = ' ')
			feat = np.reshape(feat, (48, 48, 1))
			datas.append((feat,int(label)))
	feats,labels = zip(*datas)
	feats, labels = prepro(feats, labels, norm, balance)

	return train_test_split(feats, labels, test_size = 0.1, random_state = 0)

def prepro(feats, labels, norm, balance):
	feats = np.asarray(feats)/255
	labels = to_categorical(np.asarray(labels, dtype = np.int32))
	
	if norm:
		o_shape = feats.shape
		feats = np.reshape(feats, (o_shape[0], -1))
		feats = preprocessing.scale(feats)
		feats = np.reshape(feats, o_shape)
	if balance:
		feats = feats
		
	return feats, labels
