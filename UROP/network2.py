import numpy as np
import tensorflow as tf
from numpy import linalg
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

""" Training of NN  """

data_x = np.loadtxt('10_outputs_data_x_100*10intervals_moreData.txt', delimiter=',')
target = np.loadtxt('10_outputs_target_100*10intervals_moreData.txt', delimiter=',')


def step_decay(epoch):
    initial_learning_rate = 0.01
    lrate = initial_learning_rate * 0.96 ** (int((epoch - 1) / 200))
    return lrate


def normalize(X, Y):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    return X, Y, np.std(X, axis=0), np.std(Y, axis=0), np.mean(X, axis=0), np.mean(Y, axis=0)


def denomalize(prediction, meany, stdy):
    prediction = prediction * stdy + meany
    return prediction


def calError(prediction, target):
    error = 0
    for i in range(len(target)):
        error = error + np.linalg.norm(prediction[i] - target[i]) / np.linalg.norm(target[i])
    error = error / len(target)
    print(error)


# print(data_x.shape)
# print(target.shape)
# print(np.std(data_x, axis=0))
# normalized_data_x, normalized_target, std_x, std_y, mean_x, mean_y = normalize(data_x, target)
train_valid_x, test_x, train_valid_y, test_y = train_test_split(data_x, target, test_size=0.2, random_state=42)
train_x, valid_x, train_y, valid_y, = train_test_split(train_valid_x, train_valid_y, test_size=0.2, random_state=41)

model = tf.keras.models.Sequential()
lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

checkpoint_filepath = 'model/model1'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=6)

model.add(tf.keras.Input(shape=2))
model.add(tf.keras.layers.Dense(10, activation='sigmoid', activity_regularizer=regularizers.l2(1e-4)))
# model.add(tf.keras.layers.Dense(20, activation='relu', activity_regularizer=regularizers.l2(1e-4)))
# model.add(tf.keras.layers.Dense(10, activation='sigmoid', activity_regularizer=regularizers.l2(1e-4)))
model.add(tf.keras.layers.Dense(20, activation='relu', activity_regularizer=regularizers.l2(1e-4)))
model.add(tf.keras.layers.Dense(10, activation='sigmoid', activity_regularizer=regularizers.l2(1e-4)))
model.add(tf.keras.layers.Dense(20, activation='relu', activity_regularizer=regularizers.l2(1e-4)))
model.add(tf.keras.layers.Dense(10, activation='linear', activity_regularizer=regularizers.l2(1e-4)))

model.compile(optimizer='adam', loss='mse', metrics=['MeanSquaredError'])

history=model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=20000, batch_size=20,
          callbacks=[earlystop_callback, lrate, model_checkpoint_callback])

train_predictions = model.predict(train_x)
print("training set error:")
calError(train_predictions,train_y)

test_predictions = model.predict(test_x)
print("Testing set error:")
calError(test_predictions, test_y)

#model.save("10outputs_model_100*10intervals_remove4.h5")


