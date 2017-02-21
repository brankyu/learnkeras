from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop

(x_train,y_train),(x_test,y_test) = mnist.load_data()
# data pre-processing
x_train = x_train.reshape(x_train.shape[0],-1)/255
x_test = x_test.reshape(x_test.shape[0],-1)/255
y_train = np_utils.to_categorical(y_train,nb_classes=10)
y_test = np_utils.to_categorical(y_test,nb_classes=10)

print (x_train[1].shape)
print (y_train[:3])

# build the model
model = Sequential([
    Dense(32,input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# activation the model
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model
print ('Training----------------')
model.fit(x_train,y_train,nb_epoch=2,batch_size=32)

# test model
loss,accuracy = model.evaluate(x_test,y_test)

print ('test loss',loss)
print ('test accuracy:',accuracy)