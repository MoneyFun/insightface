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
  if landmark is not None:
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
    #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

  if M is None:
    if bbox is None: #use center crop
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = kwargs.get('margin', 44)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret
  else: #do align using landmark
    assert len(image_size)==2

    #src = src[0:3,:]
    #dst = dst[0:3,:]


    #print(src.shape, dst.shape)
    #print(src)
    #print(dst)
    #print(M)
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    #tform3 = trans.ProjectiveTransform()
    #tform3.estimate(src, dst)
    #warped = trans.warp(img, tform3, output_shape=_shape)
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
	cv2.imshow("test", frame)
	key = cv2.waitKey(1)
	if key == ord('c'):
		cv2.imwrite("test.jpg", frame)
	img = get_input(frame)
	if img is None:
		continue
	# print(type(img))
	# print(img.shape)
	#
	# continue
	f2 = get_feature(img)
	dist = np.sum(np.square(f1-f2))
	# print(dist)
	sim = np.dot(f1, f2.T)
	print(sim)

