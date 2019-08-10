import sys
import model
import numpy as np
from utils import *
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(0)

def main():
	# set mode
	try:
		mode = sys.argv[1]
		assert (mode == 'dnn' or mode == 'cnn')
	except:
		print('Error: Model mode not found')
		exit()
	
	# set parameters
	# TODO choose good numbers of batch size and epoch
	batch = 32
	epoch = 20

	# load data	
	tr_feats, te_feats, tr_labels, te_labels = read_dataset()
	
	# data augmentation
	# TODO set up the parameters for 'ImageDataGenerator'
	augment_gen = ImageDataGenerator()
	origin_gen = ImageDataGenerator()
	
	# build model
	emotion_classifier = model.build_model(mode)
	
	# start training
	emotion_classifier.fit_generator(augment_gen.flow(tr_feats, tr_labels, batch_size = batch, seed = 0),
	steps_per_epoch = len(tr_feats)//batch,
	validation_data = origin_gen.flow(te_feats, te_labels, batch_size = batch, seed = 0),
	validation_steps = len(te_feats)//batch,
	epochs = epoch)
	
	# save model
	emotion_classifier.save_weights(mode+'.h5')

if __name__ == "__main__":
	main()
