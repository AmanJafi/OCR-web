import tensorflow as tf

model = tf.keras.models.load_model("handwriting_model.h5", compile=False)
model.export("saved_model")
