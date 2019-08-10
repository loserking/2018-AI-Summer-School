from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
# TODO choose a good dropout rate
dr = 0.
nb_class = 7

def build_model(mode):
	if mode == 'dnn':
		return dnn_model()
	if mode == 'cnn':
		return cnn_model()

def cnn_model():
	model = Sequential()
	# TODO make your model deeper

	# CNN
	# 1
	model.add(Conv2D(16, 2, input_shape = (48, 48, 1)))
	model.add(MaxPooling2D(pool_size = (1, 1)))
	model.add(Dropout(dr))

	# DNN
	model.add(Flatten())
	model.add(Dense(nb_class))
	model.add(Activation('softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
	
	return model

def dnn_model():
	model = Sequential()
	
	# DNN
	model.add(Flatten(input_shape = (48, 48, 1)))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dropout(dr))
	model.add(Dense(nb_class))
	model.add(Activation('softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
	
	return model
