from warnings import filters
import cv2 as cv 
import numpy as np
from skimage.feature import hog 
from sklearn.svm import LinearSVC
import tensorflow as tf 
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import glob
 
#1a
car = cv.imread("./input/p1/car.jpg")
feature_descriptor, hog_image = hog(car,cells_per_block=(2,2),channel_axis=2,visualize=True)
cv.imwrite("./output/ps7-1-a.png",hog_image)
print("feature descriptor length: " + str(feature_descriptor.shape))

#1b
X_train = []
y_train = []
for img_path in glob.glob("./input/p1/train_imgs/*.jpg"):
    image = cv.imread(img_path)
    feature_descriptor, hog_image = hog(image,cells_per_block=(2,2),channel_axis=2,visualize=True)
    X_train.append(feature_descriptor)
    if int(img_path[22]) == 0:
        y_train.append(0)
    elif int(img_path[22]) == 1:
        y_train.append(1)
X_train = np.array(X_train)
y_train = np.array(y_train)
print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))

#1c
svm = LinearSVC()
svm.fit(X_train,y_train)

#1d
window = np.zeros((96,32))
red = (0,0,255)
K = 1
for img_path in glob.glob("./input/p1/test_imgs/*.jpg"):
    image = cv.imread(img_path)
    #sliding window technique
    max_score = 0
    i_max = 0
    j_max = 0
    for i in range(image.shape[0] - window.shape[0]):
        for j in range(image.shape[1] - window.shape[1]):
            X = np.zeros((1,1188))
            window = image[i:i + window.shape[0], j:j + window.shape[1], :]
            feature_descriptor = hog(window,cells_per_block=(2,2),channel_axis=2,visualize=False,feature_vector=True)
            X[0,:] = feature_descriptor.ravel()
            scores = svm.decision_function(X)
            score = scores
            if max(score) > .6:
                max_score = score
                i_max = i
                j_max = j
    print("max score: " + str(max_score))
    if max_score > 0:
        detected_image = cv.rectangle(image,(i_max,j_max),(i_max + window.shape[1],j_max + window.shape[0]),color=red,thickness=1)
        cv.imwrite("./output/ps7-1-d-%s.png" % (K),detected_image)
    else: 
        cv.imwrite("./output/ps7-1-d-%s.png" % (K),image)
    K += 1
#2
ds_train_ = image_dataset_from_directory(
    './input/p2/train_imgs',
    labels='inferred',
    label_mode='categorical',
    image_size=[32, 32],
    batch_size=100,
    shuffle=True,
)
ds_test_ = image_dataset_from_directory(
    './input/p2/test_imgs',
    labels='inferred',
    label_mode='categorical',
    image_size=[32, 32],
    batch_size=100,
    shuffle=False,
)
model = keras.Sequential([
    layers.Conv2D(kernel_size = 5,filters=32,activation="relu",padding='same',input_shape=[32,32,3]),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Conv2D(kernel_size = 5,filters=32,activation="relu",padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Conv2D(kernel_size = 5,filters=64,activation="relu",padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(units=27,activation="relu"),
    layers.Dense(units=3,activation="softmax"),
])
model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    ds_train_,
    epochs=10,
)
model.save("./input/model")
print("PLOTTING THE FIGURE!!!!")
plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy') # validation accuracy; no validation in this example
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig("./output/ps7-2-a-2.png")

#test the model
print("TESTING THE MODEL!!!!!!")
scores = model.evaluate(ds_test_, verbose=0)
print('Accuracy on testing data: {}% \n Error on training data: {} \n'.format(scores[1], 1 - scores[1]))
print(model.predict(ds_test_))
#displaying an image with the detected label
print("DISPLAYING IMAGE WITH DETECTED LABEL")
from keras.preprocessing import image
classes = ["airplane", "automobile","truck"]
count = 1
fig,axs = plt.subplots(3,2)
# model = tf.keras.models.load_model(
#     "./input/model",
#     custom_objects=None, compile=True)
for img_path in glob.glob("./input/p2/display_imgs/*.png"):
    print("image path: " + str(img_path))
    print("count: " + str(count))
    image = cv.imread(img_path)
    # image = cv.resize(image,(256,256))
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image, verbose=0)
    print(pred)
    class_ID = np.argmax(pred)
    title = 'predicted ' + classes[class_ID]
    print(title)
    fig.add_subplot(3,2,count)
    plt.imshow(tf.squeeze(image))
    plt.axis('off')
    plt.title(title)
    count += 1
fig.savefig("./output/ps7-2-c.png")


