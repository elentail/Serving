## Toy Model CNN

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        # 28,28,1
        self.layer1 = tf.keras.Sequential(
            [layers.Conv2D(32, kernel_size=3, 
                           padding='same',activation='relu'),
             layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')])
        # 14,14,32
        self.layer2 = tf.keras.Sequential(
            [layers.Conv2D(64, kernel_size=3, 
                           padding='same',activation='relu'),
             layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')])
        # 7,7,64
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(10)

    def call(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        return self.fc(out)
        
## * from_logits=True - softmax 수행하기 전의 값을 사용 ##
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
tf_model = CNN()
tf_model.compile(loss=criterion, optimizer=optimizer)

train_epochs =20
batch_size = 1024
model_path = 'tf_cnn_model'
version = '1'
save_path = f'{model_path}/{version}'

tf_model.fit(x_train, y_train, batch_size=batch_size, epochs=train_epochs)

pred_y = tf_model.predict(x_test, batch_size=batch_size)
accuracy = np.sum(np.argmax(pred_y, axis=1) == y_test)/len(y_test)
print(f'accuracy : {accuracy:>.4f}')

tf.keras.models.save_model(tf_model, save_path)
#!tar -zcvf tf_cnn_model.tar.gz tf_cnn_model
