import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

""""
@author: Gustavo Machado
"""

#Modificando as imagens para evitar overfitting
train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True)
training_set = train_datagen.flow_from_directory('path/dataset/train_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('path/dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')


#cria a CNN de classe sequencial
cnn = tf.keras.models.Sequential()

#adiciona um layer convolucional. parametros = numero de filtros, tamanho do karnel, função de atv, dimenção das imagens
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#adiciona max_pooling. parametros: numero da pool(2x2),  poola de 2 em 2 pixels
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#segundo convolucional e max_pooling (sem a input_shape)
cnn.add(tf.keras.layers.Conv2D(filters=36, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#flettening(transforma em vetor)
cnn.add(tf.keras.layers.Flatten())

#adiciona uma camada totalmente conectada(densa). Parametros: Numer de neuronios e função de ativação
cnn.add(tf.keras.layers.Dense(units=150, activation='relu'))

#camada densa com uma unidade pois as saídas possiveis são 0 e 1 (2 classes), para multiclasses usar Softmax não sigmoid
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#optmização, função de perda e metricas e treina o modelo e avalia pelo test_set
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

training_set.class_indices

#salva modelo e arquitetura em um unico arquivo
cnn.save("model.h5", overwrite=True)
