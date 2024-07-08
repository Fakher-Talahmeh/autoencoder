import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, ReLU, BatchNormalization
import matplotlib.pyplot as plt

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=150, n_informative=10, noise=0.1, random_state=1)

# Preprocessing and splitting into train & test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=1)

# MinMax scaling
minMax_process = MinMaxScaler()
X_train = minMax_process.fit_transform(X_train)
X_test = minMax_process.transform(X_test)

# Build the autoencoder model (encoder & decoder)
inputs = X_test.shape[1]
encoder_input = Input(shape=(inputs,))
encoder_layer = Dense(inputs*2)(encoder_input)
encoder_layer = BatchNormalization()(encoder_layer)
encoder_layer = ReLU()(encoder_layer)
bottleNeck = Dense(inputs)(encoder_layer)
decoder_layer = Dense(inputs*2)(bottleNeck)
decoder_layer = BatchNormalization()(decoder_layer)
decoder_layer = ReLU()(decoder_layer)
decoder_output = Dense(inputs, activation='linear')(decoder_layer)
autoencoder = Model(inputs=encoder_input, outputs=decoder_output)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train, batch_size=20, epochs=500, validation_data=(X_test, X_test))

# Plot the losses
plt.title("The losses")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Split the encoder from the autoencoder
e2 = Model(inputs=encoder_input, outputs=bottleNeck)
e2.save('encoder.h5')

# Generate another data
X, y = make_regression(n_samples=1000, n_features=150, n_informative=10, noise=0.1, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=1)

# Reshape y_train and y_test to 2D
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Normalize X_train, X_test, y_train, y_test
process = MinMaxScaler()
X_train = process.fit_transform(X_train)
X_test = process.transform(X_test)

process2 = MinMaxScaler()
y_train = process2.fit_transform(y_train)
y_test = process2.transform(y_test)

# Define and train the SVR model
model = SVR()
model.fit(X_train, y_train.ravel())

y_predict = model.predict(X_test)
y_predict = y_predict.reshape(-1, 1)

# Inverse Transform predictions
y_predict = process2.inverse_transform(y_predict)
y_test = process2.inverse_transform(y_test)

score = mean_absolute_error(y_test, y_predict)
print(f"The error is {score}")
