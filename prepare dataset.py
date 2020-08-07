import os
from subprocess import Popen 
import cv2 as cv
from os import path
import patoolib
from yolo import YOLO


images_path = 'E:/Master/Datasets/faces dataset/Color FERET database/classes/'



# creating the output folders
os.mkdir('dataset')

# listing images folders
images_files_list = os.listdir(images_path)

# creating a face detector object
face_det_obj = YOLO()

# looping through image folders 
for subjects in images_files_list:
	subject_path = images_path + subjects 

	# creating the output class folders 
	if not path.exists('dataset/' + subjects):
            os.mkdir('dataset/' + subjects)

	# listing files in each folder
	subject_list = os.listdir(subject_path)

	# for each image
	for subject_images in subject_list:

		# # ignoring the pl (profile left) and pr (profile right images)
		# if 'pr' in subject_images or 'pl' in subject_images:
		# 	continue

		img_path = images_path + subjects + '/' + subject_images

		patoolib.extract_archive(img_path, outdir=images_path + subjects + '/')

		# # # reading the extracted image
		# img = cv.imread(images_path + subjects + '/' + subject_images[:-4])

		# cv.imwrite('dataset/' + subjects + '/' + subject_images[:-8] + '.jpg', img)

		# # # detecting the face and saving it to the new directory
		face_det_obj.detect_image(img_path=images_path + subjects + '/' + subject_images[:-4], save_path='dataset/' + subjects + '/' + subject_images[:-8] + '.jpg')
		
		# print('dataset/' + subjects + '/' + subject_images[:-8] + '.jpg')


	print('subject : ', subjects)


