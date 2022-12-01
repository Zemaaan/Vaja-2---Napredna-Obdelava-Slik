import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy
from PIL import Image


# class ImageLoaderTrain(tf.keras.utils.Sequence):
# 	def __init__(self, x_set, y_set, batch_size):
# 		# // Posalti imena slik - array[91],
# 		self.x, self.y = x_set, y_set
# 		self.batch_size = batch_size
#
# 	def __len__(self):
# 		return math.ceil(len(self.x) / self.batch_size)
#
# 	def __getitem__(self, idx):
# 		# idx = slika za NN - index
# 		# prebrati sliko - imread cv
# 		# transofrmacija
# 		# razrerzati sliko - vrnuti manje slike (return)
# 		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
# 		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#
# 		return Slike()


# zakaj nebi uporabili resize cv2? - algoritem dela bilinear interpolation
def ZmanjsajSliko(ImeSlike):
	SlikaObjekt = cv2.imread(r"C:\Users\Hrvoje\OneDrive - Univerza v Mariboru\Namizje\Faks\Napredna Obdelava Slik\Vaja 2\Train\train\\" + ImeSlike)
	bilinear_img = cv2.resize(SlikaObjekt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
	PovecanaSlike = cv2.resize(bilinear_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
	return PovecanaSlike


def PretvorivYCBCR(SlikaObjekt):
	YcbCrSlika = cv2.cvtColor(SlikaObjekt, cv2.COLOR_BGR2YCR_CB)
	return YcbCrSlika


def IzdelajYSlike(YCbCrSlika):
	(YLumaKonenta, cb, cr) = YCbCrSlika.split()
	YLumaKonenta = numpy.asarray(YLumaKonenta)
	return YLumaKonenta


def ObradiSliko(SlikaObjekt):
	PovecanaSlika = ZmanjsajSliko(SlikaObjekt)
	YCbCrSlikaObjekt = PretvorivYCBCR(PovecanaSlika)
	YSlikaObjekt = IzdelajYSlike(YCbCrSlikaObjekt)
	Matrika = numpy.asarray(YSlikaObjekt)
	Matrika = numpy.expand_dims(Matrika, axis=0)
	NovaSlika = tf.image.extract_patches(Matrika, sizes=[1, 33, 33, 1], strides=[1, 14, 14, 1], rates=[1, 1, 1, 1], padding='VALID')
	NovaSlika = numpy.reshape(NovaSlika, [1, 33, 33, -1])
	return NovaSlika

def RazreziSlike():
	img = cv2.imread('/media/hrvoje/New Volume/Napredna Obdelava Sliki - Prenosna mapa/Vaja 2/YSlikeFaktor2/t2.png')
	# print(img.shape)
	Matrika = numpy.asarray(img)
	Matrika = numpy.expand_dims(Matrika, axis=0)
	print(Matrika.shape)
	NovaSlika = tf.image.extract_patches(Matrika, sizes=[1, 33, 33, 1], strides=[1, 14, 14, 1], rates=[1, 1, 1, 1], padding='VALID')

	print(type(NovaSlika))
	print(NovaSlika.shape)
