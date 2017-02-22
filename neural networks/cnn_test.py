from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import adam
from keras import metrics


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# data pre-processing
x_train = x_train.reshape(x_train.shape[0], -1)/255
x_test = x_test.reshape(x_test.shape[0], -1)/255
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)


model = Sequential
model.add(Convolution2D(nb_filter = 32, nb_row = 5, nb_col = 5, border_mode = 'same', dim_ordering= 'tf', input_shape= (28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), border_mode = 'same'))
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), border_mode='same'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer=adam, loss='categorical-crossentroy', metrics=['accuracy'])
model.fit(x_train, y_train, nb_epoch=1, batch_size=50)

# test model
loss,accuracy = model.evaluate(x_test, y_test)





