import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# The model.json says input [null, 43, 232, 1] but weights say flatten=704
# With T=232, mels 31-38 all give 704. 
# TM audio uses 40 mel bands internally but the model.json header may be wrong.
# Let's check: what does the actual weight binary total size tell us?
import numpy as np, json

with open("tm-my-audio-model/model.json") as f:
    mj = json.load(f)
manifest = mj["weightsManifest"][0]["weights"]

total_params = 0
for w in manifest:
    n = 1
    for d in w["shape"]: n *= d
    total_params += n
    print(f"  {w['name']:30s} {str(w['shape']):20s} = {n:,}")

import os
bin_size = os.path.getsize("tm-my-audio-model/weights.bin")
print(f"\nweights.bin size: {bin_size:,} bytes = {bin_size//4:,} float32 params")
print(f"Sum from manifest: {total_params:,} params")

# Now: dense_1/kernel is [704, 2000]
# conv layers contribute: conv1=(2*8*1*8)=128, conv2=(2*4*8*32)=2048, 
#                         conv3=(2*4*32*32)=8192, conv4=(2*4*32*32)=8192
# dense_1/kernel = flatten_size * 2000 = 704*2000 = 1,408,000
# So flatten_size IS 704. The model.json input shape [null,43,232,1] is WRONG metadata.
# The actual trained input that gives flatten=704 with this arch:
# We need freq_out * time_out * 32 = 704
# freq_out * time_out = 22
# With T=232: freq_out must be 22/time_out
# Let's compute what freq_out we get for each mel count:
print("\nFreq dimension after all conv/pool layers for each mel count:")
for mels in range(28, 45):
    try:
        inp = tf.keras.Input(shape=(mels, 232, 1))
        x = tf.keras.layers.Conv2D(8,  (2,8), activation="relu", padding="valid")(inp)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(32, (2,4), activation="relu", padding="valid")(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(32, (2,4), activation="relu", padding="valid")(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(32, (2,4), activation="relu", padding="valid")(x)
        x = tf.keras.layers.MaxPooling2D((1,2))(x)
        freq_out = int(x.shape[1])
        time_out = int(x.shape[2])
        flat = freq_out * time_out * 32
        print(f"  mels={mels:2d} -> freq_out={freq_out}, time_out={time_out}, flat={flat}")
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"  mels={mels:2d} -> ERROR: {e}")
        tf.keras.backend.clear_session()
