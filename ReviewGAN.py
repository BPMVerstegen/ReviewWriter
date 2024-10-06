import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences
max_length = 200
padded_x_train = pad_sequences(x_train, maxlen=max_length)
padded_x_test = pad_sequences(x_test, maxlen=max_length)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define generator model
generator = Sequential()
generator.add(Embedding(input_dim=10000, output_dim=128, input_length=max_length))
generator.add(LSTM(128, return_sequences=True))
generator.add(Dropout(0.2))
generator.add(LSTM(128))
generator.add(Dense(128, activation='relu'))
generator.add(Dense(256, activation='relu'))
generator.add(Dense(2, activation='sigmoid'))
# Use learning_rate instead of lr
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# Define discriminator model
discriminator = Sequential()
discriminator.add(Embedding(input_dim=10000, output_dim=128, input_length=max_length))
discriminator.add(LSTM(128, return_sequences=True))
discriminator.add(Dropout(0.2))
discriminator.add(LSTM(128))
discriminator.add(Dense(128, activation='relu'))
# Change the output layer to have 2 units and softmax activation
discriminator.add(Dense(2, activation='softmax')) # This line is changed
# Use learning_rate instead of lr
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# Define GAN model
gan_input = Input(shape=(max_length,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(inputs=gan_input, outputs=gan_output)
# Use learning_rate instead of lr
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# Define training function
def train_gan(gan, generator, discriminator, x_train, y_train, epochs=10, batch_size=32):
    for epoch in range(epochs):
        for i in range(len(x_train) // batch_size):
            # Generate random noise
            noise = np.random.rand(batch_size, max_length)

            # Generate fake reviews
            fake_reviews = generator.predict(noise)

            # Generate real reviews
            real_reviews = padded_x_train[i*batch_size:(i+1)*batch_size]

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(real_reviews, y_train[i*batch_size:(i+1)*batch_size])
            d_loss_fake = discriminator.train_on_batch(fake_reviews, np.zeros((batch_size, 2)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = gan.train_on_batch(noise, y_train[i*batch_size:(i+1)*batch_size])

        print(f'Epoch {epoch+1}, Discriminator loss: {d_loss[0]}, Generator loss: {g_loss}')

# Train GAN
train_gan(gan, generator, discriminator, padded_x_train, y_train, epochs=10)

# Define function to generate reviews
def generate_reviews(prompt, length=200):
    # Convert prompt to sequence
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([prompt])
    prompt_seq = tokenizer.texts_to_sequences([prompt])[0]
    prompt_seq = pad_sequences([prompt_seq], maxlen=length)

    # Generate review
    review = generator.predict(prompt_seq)
    review = np.argmax(review, axis=1)
    return review

# Test the generator
prompt = input("Enter a text prompt: ")
review = generate_reviews(prompt)
print("Generated review:", review)
