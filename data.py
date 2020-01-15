import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator




def create_model():
	model = Sequential()
	model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
	model.add(BatchNormalization())
	model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(units = 4096, activation = 'relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Dense(units = 4096, activation = 'relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Dense(units = 1000, activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(units = 38, activation = 'softmax'))
	return model


def train_model(model):
	model.load_weights('config/weights.hdf5')
	model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	train_gen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')
	valid_gen = ImageDataGenerator(rescale=1./255)
	train_data = train_gen.flow_from_directory('dataset/train',
                                                 target_size=(224, 224),
                                                 batch_size=256,
                                                 class_mode='categorical')
	valid_data = valid_gen.flow_from_directory('dataset/valid',
                                            target_size=(224, 224),
                                            batch_size=256,
                                            class_mode='categorical')
	checkpoint = ModelCheckpoint('config/best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
	callbacks = [checkpoint]
	model.fit_generator(train_data,
                         steps_per_epoch=train_data.samples//256,
                         validation_data=valid_data,
                         epochs=25,
                         validation_steps=valid_data.samples//256,
                         callbacks=callbacks)
	model.save('config/cnn.hdf5')
	return model


def get_prediction(model, img):
	img = np.expand_dims(img, axis=0)/255
	prediction = model.predict(img)
	d = prediction.flatten()
	j = d.max()
	classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
	'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
	'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
	'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
	'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
	'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
	'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
	'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 
	'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
	'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
	'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
	'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
	'Tomato___healthy']
	for index,item in enumerate(d):
	    if item == j:
	        return classes[index]
