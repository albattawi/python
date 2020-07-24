from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))


# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "h_encoded1" is the hidden encoded representation of the encoded result
h_encoded1 = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded1 = Dense(784, activation='sigmoid')(h_encoded1)
decoded = Dense(784, activation='sigmoid')(decoded1)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded, name='encoder')
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

prediction = autoencoder.predict(x_test)
print('Predicte x_test: ',prediction)

# Display the middle layer
input_test = x_test[:1]
predictest = autoencoder.predict(input_test)
#print('Predicte x_test: ', predictest)

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(6, 3.5)
input_test_reshaped = input_test.reshape((28, 28))
reconsstruction_reshaped = predictest.reshape((28, 28))
axes[0].imshow(input_test_reshaped)
axes[0].set_title('Original image')
axes[1].imshow(reconsstruction_reshaped)
axes[1].set_title('Reconstruction')
plt.show()
from keract import get_activations, display_activations
activations = get_activations(encoder, input_test)
display_activations(activations, cmap="gray", save=False)

# visualize one test data before and after reconstructed
# and let user enter the image number
from matplotlib import pyplot as plt
print()
img = int(input('Please Enter image number: '))
plt.imshow(x_test[img].reshape(28,28))
plt.title('Original Image')
plt.show()

plt.imshow(prediction[img].reshape(28, 28))
plt.title('Reconstructed Image')
plt.show()

# compile the autoencoder to add the accuracy , because originally showing only the loss
autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the autoencoder to get histpory keys as ['loss', 'accuracy', 'val_loss', 'val_accuracy']
history = autoencoder.fit(x_test, x_test, validation_split=0.33, epochs=5, batch_size=256, verbose=0)
#print(history.history.keys())

# Draw the accuracy and loss using Matplotlib Plot
# Plot the Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()
#Plot the Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='lower right')
plt.show()