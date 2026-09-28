import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

inp = tf.keras.Input(shape=(43, 232, 1))
x = tf.keras.layers.Conv2D(8, (2,8), activation='relu', padding='valid')(inp)
print('conv2d_1:', x.shape)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
print('pool1:', x.shape)
x = tf.keras.layers.Conv2D(32, (2,4), activation='relu', padding='valid')(x)
print('conv2d_2:', x.shape)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
print('pool2:', x.shape)
x = tf.keras.layers.Conv2D(32, (2,4), activation='relu', padding='valid')(x)
print('conv2d_3:', x.shape)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
print('pool3:', x.shape)
x = tf.keras.layers.Conv2D(32, (2,4), activation='relu', padding='valid')(x)
print('conv2d_4:', x.shape)
x = tf.keras.layers.MaxPooling2D((1,2))(x)
print('pool4:', x.shape)
x = tf.keras.layers.Flatten()(x)
print('FLATTEN SIZE:', x.shape[-1])
