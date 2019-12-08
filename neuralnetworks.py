from numpy import loadtxt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD

# load the dataset
dataset = loadtxt('wine.csv', delimiter=',')
dataset2 = loadtxt('wineTest.csv', delimiter=',')

average = 0
total = ""

for i in range(20):
    # split into input (X) and output (y) Training & Test variables
    X_train = dataset[:, 1:14]
    y_train = dataset[:, 0]
    y_train = to_categorical(y_train-1)

    X_test = dataset2[:, 1:14]
    y_test = dataset2[:, 0]
    y_test = to_categorical(y_test - 1)

    # normalize
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=13, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    adam = tf.optimizers.Adam(learning_rate=0.005)

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(X_train, y_train, epochs=550, batch_size=15, verbose=1)

    # evaluate the keras model
    _, accuracy = model.evaluate(X_train, y_train)
    _, accuracy2 = model.evaluate(X_test, y_test)

    # make probability predictions with the model
    # predictions = model.predict_classes(X_train)
    # print('Accuracy TRAIN: %.2f' % (accuracy*100))
    # print('Accuracy TEST: %.2f' % (accuracy2*100))
    total += ('Accuracy TRAIN: %.2f' % (accuracy * 100))
    total += ('Accuracy TEST: %.2f' % (accuracy2*100))

    average += accuracy2

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

print('TOTAL', total)
print('AVERAGE', average/20)
def get_model():
    return model
