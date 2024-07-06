import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 18

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Dataset",
    shuffle=True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

class_names = dataset.class_names
# print(class_names)

# plt.figure(figsize=(10,10))
# for image_batch , labels_batch in dataset.take(1):
#   print(image_batch.shape)
#   print(labels_batch.numpy())
#   for i in range(12):
#     ax = plt.subplot(3,4,i+1)
#     plt.imshow(image_batch[i].numpy().astype('uint8'))
#     plt.title(class_names[labels_batch[i]])
#     plt.axis('off')

"""80% of data => Training

20% of Training => test

      10% of test data => Validation

      10% of test data => Test
"""

train_size=0.8
# len(dataset)*train_size

training_ds = dataset.take(54)
# len(training_ds)

test_ds = dataset.skip(54)
# len(test_ds)

val_size = 0.1
# len(dataset)*val_size

val_ds = test_ds.take(6)
# len(val_ds)

test_ds = test_ds.skip(6)
# len(test_ds)

def get_dataset_partition_ds(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):

  # Calculate dataset sizes
  ds_size = len(list(ds))
  train_size = int(train_split * ds_size)
  val_size = int(val_split * ds_size)
  test_size = int(test_split * ds_size)

  # Partition the dataset
  training_ds = ds.take(train_size)
  val_ds = ds.skip(train_size).take(val_size)
  test_ds = ds.skip(train_size).skip(val_size)

  # Shuffle the datasets
  if shuffle:
    training_ds = training_ds.shuffle(shuffle_size)
    val_ds = val_ds.shuffle(shuffle_size)
    test_ds = test_ds.shuffle(shuffle_size)

  return training_ds, val_ds, test_ds

training_ds, val_ds , test_ds = get_dataset_partition_ds(dataset)

training_ds = training_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# for image_batch, labels_batch in dataset.take(1):
#   print(image_batch[0].numpy()/255)

resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2)
])

"""# **Model training and building CNN model**"""

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 2
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    training_ds,
    batch_size=BATCH_SIZE,
    validation_data = val_ds,
    verbose=1,
    epochs=EPOCHS
)

scores = model.evaluate(test_ds)

# print(scores)

# print(history.params)
# print(history.history.keys())

# print(history.history['loss'])

# print(history.history['accuracy'])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# plt.figure(figsize=(8,8))
# plt.subplot(1, 2, 1)
# plt.plot(range(EPOCHS), acc, label="Training Accuracy")
# plt.plot(range(EPOCHS), val_acc, label="Validation Accuracy")
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(range(EPOCHS), loss, label='Training Loss')
# plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title("Training and Validation Loss")
# plt.show()


# for images_batch, labels_batch in test_ds.take(1):
#   first_image = images_batch[0].numpy().astype('uint8')
#   print("First Image to Predict")
#   plt.imshow(first_image)
#   print("Actual Label:", class_names[labels_batch[0].numpy()])
#   batch_prediction = model.predict(images_batch)
#   print("Predicted Label:", class_names[np.argmax(batch_prediction[0])])

# def predict(model,img):
#   img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#   img_array = tf.expand_dims(img_array, 0) #Create a batch

#   predictions = model.predict(img_array)

#   predicted_class = class_names[np.argmax(predictions[0])]
#   confidence = round(100 * (np.max(predictions[0])), 2)
#   return predicted_class, confidence

# plt.figure(figsize=(15,15))
# for images, labels in test_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3,3,i+1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.show()
#     predicted_class, confidence = predict(model, images[i].numpy())
#     actual_class = class_names[labels[i]]
#     plt.title(f"Actual : {actual_class}, \n Predicted: {predicted_class}. \n Confidence: {confidence}%")
#     plt.axis('off'){h

model.save("models/my_model.keras")