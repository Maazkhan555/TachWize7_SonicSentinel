import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

found = []
for mels in range(20, 60):
    for T in range(100, 300):
        try:
            inp = tf.keras.Input(shape=(mels, T, 1))
            x = tf.keras.layers.Conv2D(8, (2,8), activation='relu', padding='valid')(inp)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            x = tf.keras.layers.Conv2D(32, (2,4), activation='relu', padding='valid')(x)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            x = tf.keras.layers.Conv2D(32, (2,4), activation='relu', padding='valid')(x)
            x = tf.keras.layers.MaxPooling2D((2,2))(x)
            x = tf.keras.layers.Conv2D(32, (2,4), activation='relu', padding='valid')(x)
            x = tf.keras.layers.MaxPooling2D((1,2))(x)
            x = tf.keras.layers.Flatten()(x)
            flat = x.shape[-1]
            if flat == 704:
                found.append((mels, T))
        except:
            pass
        tf.keras.backend.clear_session()

print('Found (mels, T):', found[:20])
