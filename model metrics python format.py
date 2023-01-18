########################
### model parameters ###
########################


#1. CV2 Image Preprocessing

#1a: Image Resizer
cv2.resize(
		dsize = (32,32),
		interpolation=cv2.INTER_CUBIC)

#1b: Cropping Image
img[y=2:h=30, x=2:w=30]
#i.e. this is cropping out 2 pixels from each side, keeping the middle 28x28 pixels out of 32x32

#1c: Blending With A Second Image
cv2.addWeighted(
			data1=img,
			alpha=1.5,
			data2=np.zeros(img.shape, img.dtype),
			beta=.5,
			gamma=0)


#1d: Inverting Colors
cv2.bitwise_not(img)


#1e: Converting To Grayscale
img/255   





#2. Sequential() Neural Network Classifier Model
model = Sequential()

#2a: Convolution Layer
model.add(Conv2D(
			  filters=32,
			  kernel_size=(3, 3),
			  padding = 'same',
			  activation = 'relu',
			  kernel_initializer='he_uniform',
			  input_shape=(28, 28, 3)))


#2c: Input Window Size For Taking Maximum Value In Downsampling
model.add(MaxPooling2D((2, 2)))


#2d: Flatten The Image
model.add(Flatten()) #no params specified


#2e: Densely-Connected Neural Network Layers

model.add(Dense(
			 units=128,
			 activation='relu',
			 kernel_initializer='he_uniform'))

model.add(Dense(
			 units=1,
			 activation='sigmoid'))


#2f: Model Compilation
model.compile(
		    optimizer=SGD(lr=.001, momentum=.9),
		    loss='binary_crossentropy',
		    metrics='accuracy')


#2g: Model Fit
model.fit(epochs = 15)




#3. Decision Line for considering something a flip opportunity ("1" label): .5
#i.e. if predicted probability of it being a flip opportunity is >= .5 (standard rounding),
#consider it a flip opportunity
