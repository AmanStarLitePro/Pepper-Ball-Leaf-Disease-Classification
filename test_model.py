import tensorflow as tf

new_model = tf.keras.models.load_model("models/my_model.keras")
print(new_model.summary())