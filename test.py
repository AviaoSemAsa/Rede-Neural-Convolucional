from keras import models
import numpy as np
import keras.utils as image

#carrega modelo
cnn = models.load_model('model.h5')

test_images=['test_images\cat_or_dog_1.jpg',
             'test_images\cat_or_dog_2.jpg',
             'test_images\cat_or_dog_3.jpg',
             'test_images\cat_or_dog_4.jpg']

#carrega cada imagem da lista, converte para array e corrige as dimenções
for i in test_images:
  test_image = image.load_img(i, target_size = (64, 64))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  result = cnn.predict(test_image)

  if result[0][0] == 1:
    prediction = 'dog'
  else:
    prediction = 'cat'

  print(prediction)