import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread('test.jpg')
img = model.get_input(img)
f1 = model.get_feature(img)
print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
cap = cv2.VideoCapture(0)
while True:
	ret,frame = cap.read()
	# print(frame.shape)
	cv2.imshow("test", frame)
	key = cv2.waitKey(1)
	if key == ord('c'):
		cv2.imwrite("test.jpg", frame)
	img = model.get_input(frame)
	if img is None:
		continue
	# print(type(img))
	# print(img.shape)
	#
	# continue
	f2 = model.get_feature(img)
	dist = np.sum(np.square(f1-f2))
	# print(dist)
	sim = np.dot(f1, f2.T)
	print(sim)

