import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy
from PomocneFunkcije import ObradiSliko
from skimage.io import imread


class ImageProcessorLoader(tf.keras.utils.Sequence):

	def __init__(self, x_set, y_set, batch_size):  # (x_set) = putanja do KoncnaUcnaZbirka, y_set = tensorji slik
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		return numpy.array([ObradiSliko(imread('KoncnaUcnaZbirka/Faktor2/' + file_name)) for file_name in batch_x]), numpy.array([ObradiSliko(imread('KoncnaUcnaZbirka/Faktor2/' + file_name)) for file_name in batch_x])
		# return numpy.array([ObradiSliko(imread(file_name)) for file_name in batch_x]), numpy.array(batch_y)


# 33.6
# TrainDataset = tf.keras.utils.image_dataset_from_directory('Train', shuffle=False)
# TestDataset = tf.keras.utils.image_dataset_from_directory('Test', shuffle=False)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=(33, 33, 1)),
	tf.keras.layers.Conv2D(64, (9, 9), activation=tf.nn.relu),
	tf.keras.layers.Conv2D(32, (5, 5), activation=tf.nn.relu),
	tf.keras.layers.Conv2D(1, (5, 5), activation=tf.nn.relu)
])

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
print(model.summary())
model.build()
model.fit(ImageProcessorLoader(os.listdir('KoncnaUcnaZbirka/Faktor2'), os.listdir('KoncnaUcnaZbirka/Faktor2'), 1))

# piksli so razred

# ZmanjsajSliko(r'C:\Users\Hrvoje\OneDrive - Univerza v Mariboru\Namizje\Faks\Napredna Obdelava Slik\Vaja 2\birb.jpeg', 2)
# SeznamDatotek = os.listdir(r'C:\Users\Hrvoje\OneDrive - Univerza v Mariboru\Namizje\Faks\Napredna Obdelava Slik\Vaja 2\YcbCr\Faktor2\\')



# model.fit()
