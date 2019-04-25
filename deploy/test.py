import argparse
import cv2
import sys
import numpy as np
import mxnet as mx
import os
from mtcnn_detector import MtcnnDetector
from skimage import transform as trans
import sklearn
from sklearn.decomposition import PCA
import math

def preprocess(img, bbox=None, landmark=None, **kwargs):
  M = None
  image_size = []
  str_image_size = kwargs.get('image_size', '')
  if len(str_image_size)>0:
    image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size)==1:
      image_size = [image_size[0], image_size[0]]
    assert len(image_size)==2
    assert image_size[0]==112
    assert image_size[0]==112 or image_size[1]==96

  assert len(image_size)==2
  src = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041] ], dtype=np.float32 )
  if image_size[1]==112:
    src[:,0] += 8.0
  dst = landmark.astype(np.float32)

  tform = trans.SimilarityTransform()
  tform.estimate(dst, src)
  M = tform.params[0:2,:]

  assert len(image_size)==2

  warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

  return warped

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model


def get_input(face_img):
	ret = detector.detect_face(face_img, det_type = args.det)
	if ret is None:
	  return None
	bbox, points = ret
	if bbox.shape[0]==0:
	  return None
	bbox = bbox[0,0:4]
	points = points[0,:].reshape((2,5)).T
	#print(bbox)
	#print(points)
	nimg = preprocess(face_img, bbox, points, image_size='112,112')
	nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
	aligned = np.transpose(nimg, (2,0,1))
	return aligned

def get_feature(aligned):
	input_blob = np.expand_dims(aligned, axis=0)
	data = mx.nd.array(input_blob)
	db = mx.io.DataBatch(data=(data,))
	model.forward(db, is_train=False)
	embedding = model.get_outputs()[0].asnumpy()
	embedding = sklearn.preprocessing.normalize(embedding).flatten()
	return embedding


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

ctx = mx.cpu()
_vec = args.image_size.split(',')
assert len(_vec)==2
image_size = (int(_vec[0]), int(_vec[1]))
model = get_model(ctx, image_size, args.model, 'fc1')

det_threshold = [0.6,0.7,0.8]

mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)

img = cv2.imread('test.jpg')
img = get_input(img)
f1 = get_feature(img)
print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
cap = cv2.VideoCapture(0)
while True:
	ret,frame = cap.read()
	# print(frame.shape)

	ret = detector.detect_face(frame)
	if ret is None:
		cv2.imshow("test", frame)
		cv2.waitKey(1)
		continue

	bboxes, points = ret
	if bboxes.shape[0]==0:
		cv2.imshow("test", frame)
		cv2.waitKey(1)
		continue

	for bbox,point in zip(bboxes, points):
		# print("bbox[4] = ", bbox[4])


		if bbox[4] > 0.9 and (bbox[3] -bbox[1] > 100):
			# print(bbox.shape)
			# print(point.shape)

			bbox = bbox[0:4]
			point = point.reshape((2,5)).T
			for p in point:
				cv2.circle(frame, tuple(p), 1, (0, 0, 255), 4)
			# cv2.circle(frame, tuple(point[4]), 1, (0, 0, 255), 4)

			z_eye = math.atan2(point[1][1] - point[0][1], point[1][0] - point[0][0])
			z_mouth = math.atan2(point[4][1] - point[3][1], point[4][0] - point[3][0])
			z = math.degrees((z_eye + z_mouth)/2)
			# print(math.degrees(z))
			x_eye_left = point[0][0] - bbox[0]
			x_eye_right = bbox[2] - point[1][0]

			x_mouth_left = point[3][0] - bbox[0]
			x_mouth_right = bbox[2] - point[4][0]

			if x_eye_left > 0 and x_eye_right > 0 and x_mouth_left > 0 and x_mouth_right > 0:
				# print("x_eye_left = ", x_eye_left)
				# print("x_eye_right = ", x_eye_right)

				# print("x_mouth_left = ", x_mouth_left)
				# print("x_mouth_right = ", x_mouth_right)
				x_eye = math.asin(1.0 * (x_eye_left - x_eye_right)/(x_eye_left + x_eye_right))
				x_mouth = math.asin(1.0 * (x_mouth_left - x_mouth_right)/ (x_mouth_left + x_mouth_right))
				x = math.degrees((x_eye + x_mouth) / 2)
				# print(x)


			y_up = point[0][1] + point[1][1] + point[2][1] - bbox[1] * 3
			y = math.asin((y_up / (bbox[3] - bbox[1]))/1.4 - 1)
			y = math.degrees(y * 2)
			print(y)

			nimg = preprocess(frame, bbox, point, image_size='112,112')
			cv2.imshow("face", nimg)
			cv2.waitKey(1)

			nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
			img = np.transpose(nimg, (2,0,1))

			f2 = get_feature(img)
			dist = np.sum(np.square(f1-f2))
			# print(dist)
			sim = np.dot(f1, f2.T)
			# print("sim = ", sim)
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])), (255,0,0), 2)

	cv2.imshow("test", frame)
	key = cv2.waitKey(1)
	if key == ord('c'):
		cv2.imwrite("test.jpg", frame)
