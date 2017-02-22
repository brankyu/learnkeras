from keras.datasets import mnist
from keras.layers import SimpleRNN,Activation,Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
CELL_SIZE = 50
OUTPUT_SIZE = 10
LR = 0.001

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(28, 28, -1)/255
x_test = x_test.reshape(28, 28, -1)/255
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

# RNN cell
model = Sequential
model.add(SimpleRNN(
    batch_input_shape = (None, TIME_STEPS, INPUT_SIZE),
    output_dim = CELL_SIZE,
    unroll = True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# training
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    x_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(x_batch, y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)





