import sys, cv2, dlib, os, math, scipy.ndimage
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from PIL import Image

def shape_to_np(shape, dtype="int"):

	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def triangulation(landmarks):

	tri = Delaunay(landmarks)
	# The result is a list of triangles represented by the indices of landmark points
	return tri

def plot(path, landmarks):

	# Apply Delaunay Triangulation
	tri = triangulation(landmarks)

	if tri is not None:
		# if len(landmarks) == 0:
		#     print("No face found in", path)
		# else:
		#     landmarks = stasm.force_points_into_image(landmarks, img)
		#     for point in landmarks:
		#         img[int(round(point[1]))][int(round(point[0]))] = 0

		# cv2.imshow("stasm minimal", img)
		# cv2.waitKey(0)

		# Read original image in RGB
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# add image to the plot
		plt.imshow(img)
		# add triangles to the plot
		plt.triplot(landmarks[:,0], landmarks[:,1], tri.simplices.copy(), 'w-')
		# add landmarks to the plot
		plt.plot(landmarks[:,0], landmarks[:,1], 'm.')
		# show the plot and close it
		plt.show()
		# plt.savefig(os.path.splitext(path)[0]+'_tri'+os.path.splitext(path)[1])
		plt.close()
	else:
		print("Delaunay Triangulation failed on", path)

def landmarkPredictor(path, detector, predictor):

	# Read original image in grayscale
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	if img is None:
	    print("Cannot load", path)
	    raise SystemExit

	## Use stasm to predict landmark
	# landmarks = stasm.search_single(img)

	## Use dlib to predict landmark
	# detect faces in the grayscale image
	rects = detector(img, 1)

	if not rects:
		return None
	# Only predict the landmarks for the first face
	landmarks = predictor(img, rects[0])
	landmarks = shape_to_np(landmarks)

	return landmarks

def image2landmarks(imageDir):

	# directory of images
	directory = os.fsencode(imageDir)

	# dlib initializing
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	# Go through everything in the directory
	for file in os.listdir(directory):
		
		filename = os.fsdecode(file)
		
		if filename.endswith(".jpg") or filename.endswith(".png"):
			
			# path to the image file
			path = imageDir+filename
			# get landmarks from predictor
			landmarks = landmarkPredictor(path, detector, predictor)
			
			if landmarks is not None:
				f = open(path, 'rb')
				width, height = Image.open(f).size
				width = width - 1
				height = height - 1

				# Add the corners of the image and half way points between those corners as corresponding points as well
				landmarks = np.append(landmarks, [[0, 0], [0, math.floor(width/2)], [0, width], [math.floor(height/2), 0], [math.floor(height/2), width], [height, 0], [height, math.floor(width/2)], [height, width]], axis=0)
				
				# save the landmarks to seperated txt files
				np.savetxt(os.path.splitext(path)[0]+'.txt', landmarks, fmt='%d')
			else:
				print("Landmark Prediction failed on", path)
			continue
		else:
			continue

# Visualize the result of Triangulation
def plotTriangulation(imageDir):

	directory = os.fsencode(imageDir)

	for file in os.listdir(directory):

		filename = os.fsdecode(file)

		if filename.endswith(".jpg") or filename.endswith(".png"):
			
			# path to the image file
			path = imageDir+filename
			# load landmarks from txt files
			landmarks = np.loadtxt(os.path.splitext(path)[0]+'.txt')          
			# print(landmarks)
			if landmarks is not None:
				plot(path, landmarks)
			else:
				print("Landmark Prediction failed on", path)
			continue
		else:
			continue

# Calculate the affine transformation matrix for each triangle pair
def getAffineMatrice(src_tri, dest_tri):

	amat = np.dot(src_tri, np.linalg.inv(dest_tri))[:2, :]
	return amat

def warp_to(src_img, src_landmarks, dest_landmarks):

	width, height, colors = src_img.shape

	# initialize the array to store the resulted image
	result_img = np.zeros((height, width, 3), np.uint8)

	# construct an array of all coordinates in the resulted image
	coor = np.asarray([(x, y) for y in range(0, height)
                     for x in range(0, width)], np.uint8)
	
	# triangulation on morphed image
	dest_tri = triangulation(dest_landmarks)

	ones = [1, 1, 1]
	
	# find the triangle each coordinate, in the resulted image, belongs to 
	tri_index = dest_tri.find_simplex(coor)

	# go through each triangle
	for i in range(len(dest_tri.simplices)):
		# triangle indices
		tri = dest_tri.simplices[i]
		# points within the triangle
		points = coor[tri_index == i]
		# number of points within the triangle
		count = len(points)
		# get affine matrice for this pair of triangle
		amat = getAffineMatrice(np.vstack((src_landmarks[tri, :].T, ones)), np.vstack((dest_landmarks[tri, :].T, ones)))
		# use the affine matrice to transform every pixel inside the triangle to the morphed image
		morphed = np.dot(amat, np.vstack((points.T, np.ones(count))))
		# coordinates in the original image
		x, y = points.T
		# coordinates in the morphed image
		u, v = np.int32(morphed)
		# copy the color of pixels
		result_img[y, x] = src_img[v, u]

	return result_img

def warp_images(src_path, dest_path):

	# saving path
	save_path = os.path.splitext(src_path)[0]+'_'+filename(dest_path)+'/'
	save_ext = os.path.splitext(src_path)[1]

	# number of images to generate, including two original images and morphed images
	frames = 60

	if not os.path.exists(save_path):

		os.makedirs(save_path)

		# read in images
		src_img = scipy.ndimage.imread(src_path)[:, :, :3]
		dest_img = scipy.ndimage.imread(dest_path)[:, :, :3]

		# predict landmarks for both images
		src_landmarks = np.loadtxt(os.path.splitext(src_path)[0]+'.txt')
		dest_landmarks = np.loadtxt(os.path.splitext(dest_path)[0]+'.txt')

		# save the first image (original one)
		scipy.misc.imsave(save_path+int2filename(0)+save_ext, src_img)

		for x in range(1, frames-1):
			alpha = x/(frames-1)

			# landmarks for the morphed image
			morphed_landmarks = (src_landmarks * (1 - alpha) + dest_landmarks * alpha)

			s2m = warp_to(src_img, src_landmarks, morphed_landmarks)
			d2m = warp_to(dest_img, dest_landmarks, morphed_landmarks)

			# blend two warped images into one
			morphed_img = np.uint8(np.mean(np.array([s2m *  (1 - alpha), d2m * alpha]), axis=(0)) * 2)

			# save the morphed image
			scipy.misc.imsave(save_path+int2filename(x)+save_ext, morphed_img)

		# save the last image (original one)
		scipy.misc.imsave(save_path+int2filename(frames)+save_ext, dest_img)

	else:
		print(save_path, " already exists. Skip this one.")

	return save_path

# http://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html
def images2video(dir_path, save_path):

	output = save_path + "out.mp4"
	images = []

	if isinstance(dir_path, list):
		for d in dir_path:
			for f in os.listdir(d):
				if f.endswith(".jpg") or f.endswith(".png"):
					images.append(os.path.join(d, f))
	else:
		for f in os.listdir(dir_path):
			if f.endswith(".jpg") or f.endswith(".png"):
				images.append(os.path.join(dir_path, f))
		
	# Determine the width and height from the first image
	frame = cv2.imread(images[0])
	cv2.imshow('video',frame)
	height, width, channels = frame.shape

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

	for image in images:

	    frame = cv2.imread(image)

	    out.write(frame) # Write out frame to video

	    cv2.imshow('video',frame)
	    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
	        break

	# Release everything if job is finished
	out.release()
	cv2.destroyAllWindows()

def int2filename(n):
	return '{0:06d}'.format(n)

def filename(path):
	return os.path.splitext(os.path.basename(path))[0]

def main(argv):

	imageDir = './csFaculty/'
	# image2landmarks(imageDir)
	# plotTriangulation(imageDir)

	mList = []
	previousFile = ''

	# Go through every images to make a video
	for f in os.listdir(imageDir):
		if f.endswith(".jpg") or f.endswith(".png"):
			if previousFile:
				mList.append(warp_images(imageDir+previousFile, imageDir+f))
			previousFile = f

	images2video(mList, imageDir)
	
	# imgdir = warp_images(imageDir+'bala_0.jpg', imageDir+'mahe.jpg')
	# images2video(imgdir)

	# plt.imshow(img)
	# plt.show()

	# img = cv2.imread(imageDir+'mahe.jpg', cv2.IMREAD_COLOR)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# plt.imshow(img)
	# plt.show()

if __name__ == "__main__":
	# execute only if run as a script
	main(sys.argv)