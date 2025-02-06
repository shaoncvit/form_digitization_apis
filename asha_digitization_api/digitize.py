#Importing important libraries

import torch
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
import easyocr
from nltk.translate.bleu_score import sentence_bleu
import random
import os
import subprocess
import glob
import json
from natsort import natsorted
import requests
import base64
import shutil
import time
import re
import argparse
import zipfile
import rarfile


class asha_digitized:
	def __init__(self):
		self.saved_masked_image_dir = "masked_output"
		self.dir_align_image_version_1 = "aligned_version1"
		self.dir_align_image = "aligned_images"
		self.dir_template_image_info = "form_template_info"
		self.temp_folder = "temp"
		self.extra_image_folder = "extra"
		self.dir_result_small = "small_result"
		self.dir_result = "result"
		self.dir_user_upload = "upload_forms"


	#function for remove the background
	def backgroundRemoval(self, input_image_path):
	    input_image = cv2.imread(input_image_path)
	    if input_image.shape[1] >= 2000 or input_image.shape[0]>= 2000:
	        print(f"The width and height of the image is :{input_image.shape[1]},{input_image.shape[0]}")
	        print("We need to resize")
	        input_image = cv2.resize(input_image, (input_image.shape[1]//2, input_image.shape[0]//2))
	        cv2.imwrite(input_image_path, input_image)
	    # dir_masked_image = "masked_output"
	    #remove the background
	    # self.backgroundRemoval(input_image_path, self.saved_masked_image_dir)
	    python_command_backremoval = f"python U-2-Net/u2net_test.py --input_image {input_image_path} --saved_output {self.saved_masked_image_dir}"

	    subprocess.call(python_command_backremoval, shell = True)

    #function for getting the maximum contour


	def detectMaxContour(self, input_image_path, saved_masked_image_dir):
	    
	    largest_contour = None
	    max_area = 0
	    input_image_name = os.path.basename(input_image_path).split(".")[0]
	    masked_image_name = input_image_name + "_masked" + ".png"
	    masked_image_directory = os.path.join(saved_masked_image_dir, masked_image_name)

	    # Load the masked image
	    masked_image = cv2.imread(masked_image_directory)
	    if masked_image is None:
	        print(f"Failed to load masked image from {masked_image_directory}")
	        return
	    
	    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
	    largest_contour_image = masked_image.copy()

	    # Find contours
	    contours, _ = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	    if not contours:
	        print("No contours found")
	        return

	    # Find the largest contour
	    for contour in contours:
	        area = cv2.contourArea(contour)
	        if area > max_area:
	            max_area = area
	            largest_contour = contour

	    if largest_contour is None:
	        print("No largest contour found")
	        return

	    # Draw the largest contour
	    cv2.drawContours(largest_contour_image, [largest_contour], -1, (0, 255, 0), 2)

	    # Save the image with the largest contour
	    masked_image_contour_name = input_image_name + "_masked_contour" + ".jpg"
	    masked_image_contour_dir = os.path.join(self.extra_image_folder, masked_image_contour_name)

	    # Ensure the directory exists
	    os.makedirs(self.extra_image_folder, exist_ok=True)

	    success = cv2.imwrite(masked_image_contour_dir, largest_contour_image)
	    if success:
	        print(f"Saved image with contour to {masked_image_contour_dir}")
	    else:
	        print(f"Failed to save image to {masked_image_contour_dir}")


	    return largest_contour


	#getting the corner points from the contour

	def getCornerPoints(self, largest_contour):
	    hull = cv2.convexHull(largest_contour)
	    
	    epsilon = 0.04 * cv2.arcLength(hull, True)
	    approx_polygon = cv2.approxPolyDP(hull, epsilon, True)
	    
	    corner_points = approx_polygon.reshape(-1, 2)
	    
	    if len(corner_points) == 4:
	        desired_points = corner_points.copy()
	    else:
	        print("Corner points not detected. Adjusting points to have four corners.")
	        
	        # Handle cases where the number of points is not equal to four
	        if len(corner_points) > 4:
	            # If more than four points, select the four points that form the largest quadrilateral
	            desired_points = cv2.convexHull(corner_points)[:4]
	        else:
	            # If less than four points, create a rectangle or square from the available points
	            # Duplicate points to create four points
	            desired_points = np.zeros((4, 2), dtype=np.float32)
	            for i in range(4):
	                desired_points[i] = corner_points[i % len(corner_points)]

	    # Ensure points are ordered correctly for further processing
	    desired_points[[1, 3]] = desired_points[[3, 1]]
	    print(desired_points)

	    return desired_points


    #Create the combination of four points
	def createCombination(self, listPoints):
	    combinations = []
	    for i in range(len(listPoints)):
	        combination = np.concatenate((listPoints[i:], listPoints[:i]), axis = 0)
	        combinations.append(combination)

	    return combinations


	#Check the ocr working status

	def check_text_key(self, response):
	    for item in response:
	        if "text" not in item:
	            
	            return 0

	        else:
	            return 1

    #Calling the OCR-api


	def bhashini_ocr(self, image_paths, ocr_lang, modality = "handwritten", version = "v3"):
	    # sample_image = image_path

	    url = "https://ilocr.iiit.ac.in/ocr/infer"
	    base64_image = []

	    for sample_image in image_paths:

	        word_image1 = base64.b64encode(open(sample_image, 'rb').read()).decode()

	        base64_image.append(word_image1)

	    payload = json.dumps({ "modality": modality, 
	                          "language": ocr_lang, 
	                          "version": version, 
	                          "imageContent": base64_image})
	    headers = { 'Content-Type': 'application/json'}

	    response = requests.post(url, headers=headers, data=payload) 
	    ocr_output = response.json()


	    if self.check_text_key(ocr_output):
	        return ocr_output

	        # print(ocr_output)

	    else:
	        return 0


	#modifying the line of crafted output

	def modifyLine(self, bbox_text_file):
	    with open(bbox_text_file, "r") as file:
	        lines = file.readlines()
	        lines = [line.strip() for line in lines]

	        new_lines = [item for item in lines if item]

	    #write the modify lines to a new file
	    with open(bbox_text_file, "w") as file:
	        file.write("\n".join(new_lines))

	#reading the text file od detected bounding boxes
	def readfile(self, textfile):
	    with open (textfile,"r") as file:
	        lines = file.readlines()
	        return lines

	def checkDistanceHw(self, y1, y2, thresh):
	    if y2- y1 <=thresh:
	        return 1
	    else:
	        return 0
	#merging the bounding boxes

	def mergeBoundingBoxHw(self, list_bounding_box):
	    dictMergeBox = {}
	    y_coordinates_pt = []
	    mergeBox = []
	    count = 0
	    for i in range(len(list_bounding_box)-1):
	        # print(i)
	        if len(y_coordinates_pt) == 0:

	            four_points1 = list_bounding_box[i].split(",")
	            four_points2 = list_bounding_box[i+1].split(",")

	            four_points1 = [eval(k) for k in four_points1]
	            four_points2 = [eval(j) for j in four_points2]

	            points1_y = four_points1[1]
	            points2_y = four_points2[1]
	            y_coordinates_pt.append(points1_y)
	            mergeBox.append(four_points1)
	        else:
	            points1_y = y_coordinates_pt[-1]
	            four_points2 = list_bounding_box[i+1].split(",")
	            four_points2 = [eval(j) for j in four_points2]

	            points2_y = four_points2[1]
	            # print(i)


	        if self.checkDistanceHw(points1_y, points2_y, 20) == 1:
	            y_coordinates_pt.append(points2_y)
	            mergeBox.append(four_points2)
	            # print(mergeBox)

	        elif self.checkDistanceHw(points1_y, points2_y, 20) == 0:
	            if len(mergeBox) >1:
	                mergeBox.sort()
	            # print(mergeBox)
	            dictMergeBox[count+1] = mergeBox
	            y_coordinates_pt = []
	            mergeBox = []
	            y_coordinates_pt.append(points2_y)
	            mergeBox.append(four_points2)
	            count += 1
	    if len(mergeBox)>1:
	        mergeBox.sort()
	    dictMergeBox[count+1] = mergeBox

	    # print(i)
	    return dictMergeBox

	#Converting the dict of boxes to the list of boxes

	def getListBox(self, dict_boxes):
	    boxes_coord = []
	    for key , value in dict_boxes.items():
	        single_line_boxes = dict_boxes[key]
	        # print(single_line_boxes)
	        

	        for i in single_line_boxes:
	            boxes_coord.append(i)
	        


	        # boxes_coord.append([for j in len(dict_boxes[key])])
	    return boxes_coord


	#function for getting the digitized list contents which will be use for content matching technique 
	def getDigitizedList(self, img,bbox_list, lang = "bn"):
	    
	    print("For digitization ", lang)
	    digitized_word_list = []
	    count = 0
	    length_box = len(bbox_list)
	    boundingBox = []
	    start_time = time.time()
	    for i in bbox_list:

	        
	        cropped_image = img[i[1]: i[5], i[0]: i[4]]

	        #if the form's language is English
	        
	        if lang == "en":
	            ocr_reader = easyocr.Reader(["en"], gpu = True)
	            result = ocr_reader.readtext(cropped_image)
	            print(result)

	            if len(result)!= 0:
	                digitized_text = result[0][1]
	            else:
	                digitized_text = ""

	        #if the form's language is not English
	        else:

	            try:
	                if cropped_image.size == 0:
	                    raise ValueError("Image Can not be saved, Empty Image")
	                # print("Entering for Bhashini Ocr model")
	                # print("lang is :", lang)
	                image_path = str(count) + ".png"
	                image_full_path = os.path.join(self.temp_folder, image_path)

	                #saving all the cropped images to the temp foldar
	                cv2.imwrite(image_full_path, cropped_image)
	                boundingBox.append(i)
	                count += 1 

	            
	            except Exception as e:
	                print(f"Error processing Bhashini OCR for image ")


	    #Now all the cropped images are saved in to the temp foldar
	    
	    #Reading all the images and will pass it to the ocr

	    cropped_images = glob.glob(self.temp_folder +"/*.png")

	    #Ordering ascedincally based on the image names

	    cropped_images = natsorted(cropped_images)

	    #results from bhashini ocr module

	    ocr_output = self.bhashini_ocr(cropped_images, lang, "printed", "v4_robustbilingual")



	    end_time = time.time()
	    duration = end_time - start_time

	    print(f"To finish digitization of all the cropped images from one form time taken: {duration:.2f} seconds")

	    #storing digitized output and it's corresponding bounding boxes 

	    if ocr_output == 0:
	        return 0

	    else:

	        for i, out in enumerate(ocr_output):
	            print(out["text"])

	            if len(out["text"]) == 0:

	                digitized_word_list.append(("", boundingBox[i]))

	            digitized_word_list.append((out["text"], boundingBox[i]))
	            print(out["text"])

	        return digitized_word_list

	#converting the json file to list format
	def jsonTolist(self, json_file):

	    with open(json_file, "r", encoding = "utf-8")as f:
	        json_data = json.load(f)

	    #convert the json data to the desired list
	    converted_list  = [(key, value) for key, value in json_data.items()]

	    return converted_list

	#function for getting the unique characters 
	def getUniqueCharacter(self, character_list):
	    character_list = [(term.lower(), values) for term , values in character_list]

	    term_count = {}

	    for term, _ in character_list:
	        if term in term_count:
	            term_count[term] += 1
	        else:
	            term_count[term] = 1

	    character_list_unique = [(term, values) for term , values in character_list if term_count[term] == 1]
	    return character_list_unique

	#calculation of the bleu score
	def Bleu4(self, gt, pred):
	    bleu4_score = sentence_bleu([gt], pred, weights = (0.25,0.25, 0.25))

	    return bleu4_score

	#function for caluclating the bleu scores of list of words 
	def calBleu4(self, list_fields, single_word, score_hyperParam = 0.80):
	    score = []
	    count = 0

	    for word in list_fields:
	        bleu4_val = self.Bleu4(single_word, word)

	        score.append(bleu4_val)

	    if max(score)>= score_hyperParam:
	        ind = score.index(max(score))
	        count = 1

	    else:
	        ind = -1
	        count = 0

	    return ind, count

	#function for getting the matched points

	def getMatchedPoints(self, ocred_output_template, ocred_output_form):
	    digitized_form_output = [j[0].lower() for j in ocred_output_form]

	    pt_aligned_form = []

	    pt_template = []

	    main_count = 0

	    for i in ocred_output_template:
	        min_ind, count = self.calBleu4(digitized_form_output, i[0])
	        main_count += count
	        if min_ind != 1:
	            pt_template.append(i[1])
	            pt_aligned_form.append(ocred_output_form[min_ind][1])

	    return pt_template, pt_aligned_form, main_count


	#functon for converting matched points to the usable format

	def convertPointFormat(self, listPoints):
	    point_format = []
	    for i in listPoints:
	        for j in range(0, len(i),2):
	            point_format.append((i[j], i[j+1]))

	    return point_format

	#Function for clear the folder

	def clear_folder(self, folder_path):
	    # Check if the folder exists
	    if os.path.exists(folder_path):
	        # Iterate over all files in the folder and delete them
	        for filename in os.listdir(folder_path):
	            file_path = os.path.join(folder_path, filename)
	            try:
	                if os.path.isfile(file_path):
	                    os.unlink(file_path)
	                elif os.path.isdir(file_path):
	                    shutil.rmtree(file_path)
	            except Exception as e:
	                print(f"Error deleting {file_path}: {e}")
	    else:
	        print(f"Folder not found: {folder_path}")



	#function for first stage of alignment(four corner point matching)
	def getAlignVerOne(self, template_image_path, input_image_path, corner_points, four_point_track_name):

	    

	    template_image = cv2.imread(template_image_path)
	    captured_image = cv2.imread(input_image_path)
	    # captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)


	    template_image_top_left = (0,0)
	    template_image_bottom_left = (0, template_image.shape[0])
	    template_image_bottom_right = (template_image.shape[1], template_image.shape[0])

	 
	    template_image_top_right = (template_image.shape[1], 0)




	    pts1_new = np.array([template_image_top_left,template_image_bottom_left, template_image_bottom_right, template_image_top_right])
	    pts2_new = corner_points

	    h, mask = cv2.findHomography(pts2_new, pts1_new, cv2.RANSAC)

	    captured_image_warped = cv2.warpPerspective(captured_image, h, (template_image.shape[1], template_image.shape[0]))




	    os.makedirs(self.dir_align_image_version_1, exist_ok = True)

	    image_name = os.path.basename(input_image_path).split(".")[0]

	    aligned_image_name = image_name + "_alignedV1"+ ".png"

	    aligned_image_four_point_version_name = image_name + "_alignedV1_" + four_point_track_name + ".png"

	    cv2.imwrite(os.path.join(self.extra_image_folder, aligned_image_four_point_version_name), captured_image_warped)



	    cv2.imwrite(os.path.join(self.dir_align_image_version_1, aligned_image_name),captured_image_warped)


	    return os.path.join(self.dir_align_image_version_1,aligned_image_name)



	#function for final alignment phase (content matching technique)

	def finalAlign(self, aligned_imageStage1_path,template_path,pt_template, pt_form,craft_model_path =  "CRAFT-pytorch/text_detection_model/craft_mlt_25k.pth"):
	    

	    aligned_imageStage1 = cv2.imread(aligned_imageStage1_path)
	    # aligned_imageStage1 = cv2.cvtColor(aligned_imageStage1, cv2.COLOR_BGR2RGB)

	    # version_1_dirname = self.aligned_imageStage1
	    
	    version_1_imageBasename = os.path.basename(aligned_imageStage1_path).split(".")[0]


	    pt_four_point_format_template = self.convertPointFormat(pt_template)
	    pt_four_point_format_alignedV1 = self.convertPointFormat(pt_form)


	    #converting point to np.array format

	    pts1_new = np.array(pt_four_point_format_template)
	    pts2_new = np.array(pt_four_point_format_alignedV1)

	    new_h, mask = cv2.findHomography(pts2_new, pts1_new, cv2.RANSAC)

	    #reading the template image
	    template_image = cv2.imread(template_path)
	    # template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)

	    final_aligned_image = cv2.warpPerspective(aligned_imageStage1, new_h, (template_image.shape[1], template_image.shape[0]))

	    final_aligned_image_name = version_1_imageBasename + "_final.png"
	    cv2.imwrite(os.path.join(self.dir_align_image, final_aligned_image_name), final_aligned_image)

	    return os.path.join(self.dir_align_image, final_aligned_image_name)


	#function for getting the correct version of alignment

	def getCheckAlign(self,template_image_path, input_image_path, list_corner_points, template_mode):
	    flag = 0
	    print("Temporary Folder:", self.temp_folder)
	    count_four_point = 0
	    for i in range(len(list_corner_points)):
	        self.clear_folder(self.temp_folder)
	        count_four_point += 1
	        aligned_imagev1_path = self.getAlignVerOne(template_image_path, input_image_path, list_corner_points[i],str(count_four_point))
	        
	        aligned_imageStage1 = cv2.imread(aligned_imagev1_path)
	        # aligned_imageStage1 = cv2.cvtColor(aligned_imageStage1, cv2.COLOR_BGR2RGB)      


	        version_1_dirname = self.dir_align_image_version_1
	    
	        version_1_imageBasename = os.path.basename(aligned_imagev1_path).split(".")[0]

	        # craft_saved_sub_result_dir = os.path.join(version_1_dirname, version_1_imageBasename)
	        craft_saved_result_dir = os.path.join(self.temp_folder, version_1_imageBasename)
	        print("Craft saved result directory:",craft_saved_result_dir)
	        
	        os.makedirs(craft_saved_result_dir, exist_ok= True)
	    

	        #To run the craft model for detecting the contents of the form
	        #CRAFT model path for inference
	        # crafted_output = 
	        craft_model_path = "CRAFT-pytorch/text_detection_model/craft_mlt_25k.pth"
	        python_command_CRAFT = f"python CRAFT-pytorch/test.py --trained_model {craft_model_path} --test_folder {version_1_dirname} --saved_result {craft_saved_result_dir}"
	        subprocess.call(python_command_CRAFT, shell = True)

	        detected_bounding_box_subpath = "res"+ "_" + version_1_imageBasename + ".txt"

	        detected_bounding_box_fullpath = os.path.join(craft_saved_result_dir, detected_bounding_box_subpath)

	        self.modifyLine(detected_bounding_box_fullpath)

	        lines_bbox_aligned_image =self.readfile(detected_bounding_box_fullpath)
	        print("Detected bbox from the craft:",len(lines_bbox_aligned_image), flush = True)
	        dict_roi_bbox_aligned_image = self.mergeBoundingBoxHw(lines_bbox_aligned_image)
	        list_aligned_image_boxes = self.getListBox(dict_roi_bbox_aligned_image)
	        print("Bounding box format:", len(list_aligned_image_boxes), flush = True)


	        Ocred_output_form = self.getDigitizedList(aligned_imageStage1, list_aligned_image_boxes)

	        if Ocred_output_form == 0:
	            flag =0

	        #get the template image name


	        else:
	        	# template_name = os.path.basename(template_image_path).split(".")[0]
	        	
	        	template_info_dir_sub = os.path.join(self.dir_template_image_info, template_mode)
	        	template_image_name = os.path.basename(template_image_path).split(".")[0]
	        	template_info_dir = os.path.join(template_info_dir_sub,template_image_name)

	        	#get the ocred output 
	        	template_json_file = os.path.join(template_info_dir, "ocred.json")
	        	template_ocred_list = self.jsonTolist(template_json_file)

	        	#Delete the duplicate characters from the template ocred output

	        	template_unique_ocred_list = self.getUniqueCharacter(template_ocred_list)

	        	#calling the content matching function for getting the matched points
	        	pt_template, pt_form, number_of_matches = self.getMatchedPoints(template_unique_ocred_list, Ocred_output_form)

	        	print("Matches:",number_of_matches, flush = True)

	        	if number_of_matches >= 15:
	        		aligned_image_path = self.finalAlign(aligned_imagev1_path,template_image_path ,pt_template, pt_form, self.dir_align_image)
	        		flag = 1
	        		print("Sufficient matches are founded", flush = True)

	        		return (aligned_image_path, aligned_imagev1_path, number_of_matches)
	    if flag == 0:

	    	print("Sufficient matches not found", flush = True)

	    	print("Please take the clear picture and upload again", flush = True)

	    	return (aligned_imagev1_path, number_of_matches)	    

	#function extract the regions where user is interacted

	def extract_region(self, aligned_image_path, template_annotation_path, mode = "user_hand", isPlot = False):
		
		#loading the json data
		with open(template_annotation_path, "r") as file:
			annotation_data = json.load(file)

		#reading the aligned image

		aligned_image_final_version = cv2.imread(aligned_image_path)
		aligned_image_final_version_plotted = aligned_image_final_version.copy()


		if isPlot:

			#Drawing the bounding boxes
			for key, items in annotation_data.items():
				for item in items:
					bbox = item["bbox"]

					x,y,w,h = bbox["bbox_x"], bbox["bbox_y"], bbox["bbox_width"], bbox["bbox_height"]

					# Check if the key contains both "val" and "op" and the text is "user"

					if "val" in key and "op" in key and item["text"] == "user":
						cv2.rectangle(aligned_image_final_version_plotted, (x,y), (x+w, y+h), (0,255,0),2)


			#saving the plotted image

			final_aligned_image_name = os.path.basename(aligned_image_path).split(".")[0]

			image_name_with_plotted_bbox = final_aligned_image_name + "_bbox.png"

			cv2.imwrite(os.path.join(self.extra_image_folder,image_name_with_plotted_bbox), aligned_image_final_version_plotted)

		if mode == "user_hand":

			self.clear_folder(self.temp_folder)
			flag = 0

			for key, items in annotation_data.items():

				for item in items:
					bbox = item["bbox"]

					x,y,w,h = bbox["bbox_x"], bbox["bbox_y"], bbox["bbox_width"], bbox["bbox_height"]

					if item["text"] == "user":
					
						# Extracting the regions
						region = aligned_image_final_version[y:y+h, x:x+w]

						region_path = os.path.join(self.temp_folder, f"{key}.png")

						cv2.imwrite(region_path, region)
						flag = 1
			return flag



		if mode == "op_tick":
			self.clear_folder(self.temp_folder)
			flag  = 0


			for key, items in annotation_data.items():

				for item in items:

					bbox = item["bbox"]

					x,y,w,h = bbox["bbox_x"], bbox["bbox_y"], bbox["bbox_width"], bbox["bbox_height"]

					if "val" in key and "op" in key and "tick" in key:

						flag = 1

						# print("entering")
					
						# Extracting the regions
						region = aligned_image_final_version[y:y+h, x:x+w]

						region_sub_path = os.path.join(self.temp_folder, "uncroppeed_images")

						os.makedirs(region_sub_path, exist_ok = True)

						region_path = os.path.join(region_sub_path, f"{key}.png")

						cv2.imwrite(region_path, region)

						
			return flag

	#function for classify the language for detected regions
	def langClass(self, test_folder_path):
		language_classification_model_path = "langClassModel/model_epoch_0.pth"

		python_command_language_classification = f"python lang_class_inference.py --main_folder {test_folder_path} --model_path {language_classification_model_path}"

		subprocess.call(python_command_language_classification, shell = True)

	def preprocess_image(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		return binary

	def project_image(self,binary):
		projection = np.sum(binary, axis=0)
		return projection

	def find_word_boundaries(self,projection, threshold):
		word_boundaries = []
		in_word = False
		start = 0

		for i, value in enumerate(projection):
			if not in_word and value > threshold:
				in_word = True
				start = i
			elif in_word and value <= threshold:
				in_word = False
				word_boundaries.append((start, i))

		if in_word:

			word_boundaries.append((start, len(projection)))

		return word_boundaries

	def calculate_ink_percentage(self,binary, box):
		x, y, w, h = box
		region = binary[y:y+h, x:x+w]
		total_pixels = w * h
		ink_pixels = np.sum(region > 0)
		ink_percentage = (ink_pixels / total_pixels) * 100
		return ink_percentage

	def segment_words(self,bounding_boxes, ink_percentages):
		word_segments = []
		current_segment = []
	    
		for i, box in enumerate(bounding_boxes):
			if i == 0:
				current_segment.append(box)
			else:
				prev_box = bounding_boxes[i-1]
				distance = box[0] - (prev_box[0] + prev_box[2])

				if distance < 1 or distance >= 23:
					if len(current_segment) > 0:
						word_segments.append(current_segment)
					current_segment = [box]
				else:
					current_segment.append(box)

		if current_segment:
			word_segments.append(current_segment)

		return word_segments
 
	def draw_word_boxes(self,image, binary, word_boundaries):
		result = image.copy()
		bounding_boxes = []
		ink_percentages = []
		for i, (start, end) in enumerate(word_boundaries):
			box = (start, 0, end-start, image.shape[0])
			ink_percentage = self.calculate_ink_percentage(binary, box)
			ink_percentage_str = f"{ink_percentage:.2f}%"
		    
			if box[2] > 10 and float(ink_percentage_str.strip('%')) >= 13.0:
				cv2.rectangle(result, (start, 0), (end, image.shape[0]), (0, 255, 0), 2)
				bounding_boxes.append(box)
				ink_percentages.append(ink_percentage)
		return result, bounding_boxes, ink_percentages

	def save_word_segments(self,image, word_segments, image_name):
		os.makedirs(os.path.join('temp', 'cropped_images'), exist_ok=True)

		if not word_segments:
		    # Save the full image if no word segments
			cv2.imwrite(os.path.join('temp', 'cropped_images', f'{image_name}_1.jpg'), image)
		else:
			for i, segment in enumerate(word_segments):
				x_min = segment[0][0]
				x_max = segment[-1][0] + segment[-1][2]
				y_min = 0
				y_max = image.shape[0]

				cropped_image = image[y_min:y_max, x_min:x_max]
				cv2.imwrite(os.path.join('temp', 'cropped_images', f'{image_name}_{i+1}.jpg'), cropped_image)

	def draw_grouped_word_boxes(self,image, word_segments):
		result = image.copy()
		for segment in word_segments:
			x_min = segment[0][0]
			x_max = segment[-1][0] + segment[-1][2]
			y_min = 0
			y_max = image.shape[0]
			cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
		return result

	def process_image(self,image_path, predictions):
		# Read the image
		image = cv2.imread(image_path)
		image_name = os.path.splitext(os.path.basename(image_path))[0]

		# Preprocess the image
		binary = self.preprocess_image(image)

		# Project the image
		projection = self.project_image(binary)

		# Find word boundaries
		threshold = np.mean(projection) * 0.5  # Adjust this factor as needed
		word_boundaries = self.find_word_boundaries(projection, threshold)

		# Draw bounding boxes and calculate ink percentages
		result, bounding_boxes, ink_percentages = self.draw_word_boxes(image, binary, word_boundaries)

		if bounding_boxes:
			word_segments = self.segment_words(bounding_boxes, ink_percentages)
		else:
			word_segments = []

		self.save_word_segments(image, word_segments, image_name)

		prediction_value = predictions.get(os.path.basename(image_path), 0)
		return os.path.basename(image_path), len(word_segments), prediction_value

	def run_segment(self,folder_path):
		# Read the prediction file
		json_file_path = os.path.join(folder_path, "predictions.json")
		with open(json_file_path, 'r') as f:
		    predictions = json.load(f)

		# Rewrite keys to remove "temp/"
		predictions = {os.path.basename(k): v for k, v in predictions.items()}

		# Get and sort the image files
		image_files = natsorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

		final_list = []

		for image_file in image_files:
			image_path = os.path.join(folder_path, image_file)
			image_name, num_segments, prediction_value = self.process_image(image_path, predictions)
			image_name = image_name.split(".")[0]
			final_list.append((image_name, num_segments, prediction_value))

		return final_list

	def call_ocr(self, folder_path, final_list):

		image_path_ocr = os.path.join(folder_path, "cropped_images")
		image_files_ocr = natsorted([f for f in os.listdir(image_path_ocr) if f.endswith(('.png', '.jpg', '.jpeg'))])



		bengali = []

		english = []


		for info in final_list:
		    image_name = info[0]
		    # print(image_name)
		    pattern = re.compile(rf'^{image_name}(_\d+)+\.(jpg|png|jpeg)$', re.IGNORECASE)
		    # print(pattern)

		    if info[2] == 0:
		        
		        for image_file_name in image_files_ocr:
		            if pattern.match(image_file_name):
		                bengali.append(image_file_name)
		        
		    elif info[2] == 1:
		        # print("entring")
		        for image_file_name in image_files_ocr:
		            if pattern.match(image_file_name):
		                english.append(image_file_name)


		#add cropped images to each element
		cropped_file_list_english = [f'temp/cropped_images/{filename}' for filename in english]
		cropped_file_list_bengali = [f'temp/cropped_images/{filename}' for filename in bengali]


		ocr_response_english = self.bhashini_ocr(cropped_file_list_english, "en")

		# print(ocr_response_english)

		ocr_response_bengali = self.bhashini_ocr(cropped_file_list_bengali, "bn")

		# print(ocr_response_english)

		# print(ocr_response_bengali)

		#Re-arranging the ocr list

		ocred_english_merge= []

		ocred_bengali_merge = []

		# print(ocr_response_english)

		if ocr_response_english == 0:
			ocred_english_merge.append("empty")
		else:

			for i, out in enumerate(ocr_response_english):

				ocred_english_merge.append(out["text"])

		if ocr_response_bengali == 0:
			ocred_bengali_merge.append("empty")
		else:

			for j, out in enumerate(ocr_response_bengali):
				ocred_bengali_merge.append(out["text"])



		new_output_user_hand_path = {}

		for item in final_list:
			name, count, language = item
			ocred_list = ocred_english_merge if language == 1 else ocred_bengali_merge

			#check if the list has enough elements

			if len(ocred_list) >= count:
				#Take the required number of elements

				elements = ocred_list[:count]

				del ocred_list[:count]

			else:
				elements = ocred_list[:]

				ocred_list.clear()

			# Join the elements into a single string with spaces if needed

			value = ' '.join(elements) if any(len(e) > 1 for e in elements) else ''.join(elements)


			new_output_user_hand_path[name] = value


		json_result = json.dumps(new_output_user_hand_path, ensure_ascii= False, indent = 4)

		print(json_result)


		with open("small_result/user_handwritten.json", "w", encoding = "utf-8")as f:
			f.write(json_result)

	def tick_detection(self, folder_path):
	    template_image = cv2.imread("tick_template/tick.png",0)

	    input_image_path = os.path.join(folder_path, "uncroppeed_images")

	    input_image_paths = glob.glob(input_image_path + "/*")

	    input_image_paths = natsorted(input_image_paths)

	    print(input_image_paths)

	    unique_names = set()

	    accepted_tick = []

	    not_accepted_tick_2 = []

	    result_tick_detection_regions = {}





	    for path in input_image_paths:
	        # Split the path and extract the base name
	        # print(path)
	        base_name = path.split('/')[-1].split('_')[0] + '_' + path.split('/')[-1].split('_')[1]
	        unique_names.add(base_name)

	# Convert the set to a list and sort it
	    unique_names = sorted(list(unique_names))
	    print(unique_names)
	# Print the unique names
	    # print(f"Unique names: {unique_names}")


	    # Initialize a dictionary to hold lists of file paths for each unique name
	    grouped_files = {name: [] for name in unique_names}

	# Group the file paths based on the unique names
	    for path in input_image_paths:
	        base_name = path.split('/')[-1].split('_')[0] + '_' + path.split('/')[-1].split('_')[1]
	        if base_name in grouped_files:
	            grouped_files[base_name].append(path)


	    for name, paths in grouped_files.items():
	        # print(f"\nFiles for {name}:")
	        confidence_rate = []

	        for p in paths:
	            
	            input_image = cv2.imread(p)

	            input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

	            w, h, = template_image.shape[::-1]


	            res = cv2.matchTemplate(input_image_gray, template_image, cv2.TM_CCOEFF_NORMED)

	            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

	            confidence_val = max_val

	            # print(confidence_val)

	            confidence_rate.append(confidence_val)

	        max_index = confidence_rate.index(max(confidence_rate))

	        accepted_image_path =  paths[max_index]

	        image_name = os.path.basename(accepted_image_path).split(".")[0]

	        accepted_tick.append(image_name)

	        not_accepted_tick = [os.path.basename(image_name).split(".")[0] for j, image_name in enumerate(paths) if j!= max_index]

	        for k in not_accepted_tick:
	            not_accepted_tick_2.append(k)

	    # print(accepted_tick)
	    # print(not_accepted_tick_2)


	    result_tick_detection_regions["accepted"] = accepted_tick
	    result_tick_detection_regions["not_accepted"] = not_accepted_tick_2

	    result_tick_detection_json = json.dumps(result_tick_detection_regions, ensure_ascii=False, indent = 4)

	    with open("small_result/user_tick_detection.json", "w", encoding = "utf-8")as f:
	        f.write(result_tick_detection_json)

	    print(result_tick_detection_json)


	def create_final_result(self, template_json_path, small_result):



		template_json_sub_path = os.path.basename(template_json_path).split(".")[0]
		result_json_path_sub = template_json_sub_path + "_output.json"

		os.makedirs("result", exist_ok = True)

		result_json_path = os.path.join("result", result_json_path_sub)



		# Load main JSON file
		with open(template_json_path, 'r', encoding='utf-8') as file:
			main_json = json.load(file)

		# Check if handwritten JSON file exists and update main JSON
		handwritten_json_path = os.path.join(small_result, "user_handwritten.json")
		if os.path.exists(handwritten_json_path):
			with open(handwritten_json_path, 'r', encoding='utf-8') as file:
				handwritten_json = json.load(file)
			for key, value in handwritten_json.items():


				main_json[key][0]["text"] = value

		# Check if tick detection JSON file exists and update main JSON
		tick_detection_json_path = os.path.join(small_result, "user_tick_detection.json")
		if os.path.exists(tick_detection_json_path):
			with open(tick_detection_json_path, 'r', encoding='utf-8') as file:
				tick_detection_json = json.load(file)
			accepted_keys = tick_detection_json.get('accepted', [])
			not_accepted_keys = tick_detection_json.get('not_accepted', [])

			for key in not_accepted_keys:
				if key in main_json:
					del main_json[key]

		# Save the updated main JSON to a new file
		with open(result_json_path, 'w', encoding='utf-8') as file:
			json.dump(main_json, file, ensure_ascii=False, indent=4)

		print(f"Updated JSON saved to {result_json_path}")

	def extract_file(self, file_path, extract_to = "upload_forms"):

		# basename = os.path.basename(file_path).split(".")[0]

		# extract_to = os.path.join(extract_to, basename)

		# os.makedirs(extract_to, exist_ok = True)

		print(file_path)


		if file_path.endswith("zip"):
			with zipfile.ZipFile(file_path, "r") as zip_ref:
				zip_ref.extractall(extract_to)

				print(f"Extracted Zip file to {extract_to}")

		elif file_path.endswith(".rar"):
			with rarfile.RarFile(file_path, "r") as rar_ref:
				rar_ref.extractall(extract_to)

				print(f"Extracted RAR file to {extract_to}")

	def merge_final_output(self, file_name,template_name, result_path):

		if template_name == "screening_proforma":
			filenames = ["Asha_1_page1_output.json","Asha_1_page2_output.json","Asha_1_page3_output.json","Asha_1_page4_output.json"]

		elif template_name == "cbac":
			filenames = ["Cbac_1_page1_output.json","Cbac_1_page2_output.json","Cbac_1_page3_output.json","Cbac_1_page4_output.json"]


		merged_json = {}

		for i, filename in enumerate(filenames, start = 1):
			file_path = os.path.join(result_path, filename)

			key = f"page_{i}"

			if os.path.exists(file_path):
				with open(file_path, "r", encoding= "utf-8") as file:
					merged_json[key] = json.load(file)

			else:
				merged_json[key] = {}

		filename_json = file_name + ".json"
		output_file_path = os.path.join(result_path, filename_json)

		with open(output_file_path, "w", encoding = "utf-8")as out:
			json.dump(merged_json, out, ensure_ascii= False, indent = 4)

		return merged_json




	def start_engine(self, template_image_path, input_image_path):
		self.clear_folder(self.dir_align_image_version_1)
		self.clear_folder(self.temp_folder)
		self.clear_folder(self.saved_masked_image_dir)
		self.clear_folder(self.extra_image_folder)
		self.clear_folder(self.dir_result_small)
		




		os.makedirs(self.temp_folder, exist_ok = True)

		#remove the background from the image
		self.backgroundRemoval(input_image_path)

		#get the biggest contour
		biggest_contour = self.detectMaxContour(input_image_path, self.saved_masked_image_dir)

		#get the corner points from the largest contour
		corner_points = self.getCornerPoints(biggest_contour)

		#get the four combination of the corner points
		list_corner_points = self.createCombination(corner_points)

		template_mode = template_image_path.split("/")[1]

		result = self.getCheckAlign(template_image_path, input_image_path, list_corner_points, template_mode)

		if len(result) == 3:
		    final_aligned_image_path, first_stage_aligned_image_path, matches = result

		    self.clear_folder(self.temp_folder)

		    #Hardcoding the template for now

		    template_image_name = os.path.basename(template_image_path).split(".")[0]

		    template_image_dir = os.path.dirname(template_image_path)

		    template_annotation_sub_path = template_image_name + ".json"

		    template_annotation_json_path = os.path.join(template_image_dir, template_annotation_sub_path)

		    #extracting the user_handwritten regions


		    flag = self.extract_region(final_aligned_image_path, template_annotation_json_path, "user_hand")

		    if flag == 1:

		    	self.langClass(self.temp_folder)

		    	#Segmenting the user's handwritten regions
		    	track_image_segment_list = self.run_segment(self.temp_folder)
		    	print(track_image_segment_list)

		    	#call for the ocr
		    	self.call_ocr(self.temp_folder, track_image_segment_list)

		    #Extract the tick mark regions

		    flag_2 = self.extract_region(final_aligned_image_path, template_annotation_json_path, "op_tick")

		    if flag_2 == 1:

		    	self.tick_detection("temp")

		    #create the final output page wise
		    self.create_final_result(template_annotation_json_path, self.dir_result_small)


		else:

			# template_image_name = os.path.basename(template_image_path).split(".")[0]

		 #    template_image_dir = os.path.dirname(template_image_path)

		 #    template_annotation_sub_path = template_image_name + ".json"

		 #    template_annotation_json_path = os.path.join(template_image_dir, template_annotation_sub_path)
		    first_stage_aligned_image_path, matches = result
		    print(f"It is hard to align the image: {os.path.basename(input_image_path)}")
		    print("Please re capture it")
		    # self.create_final_result(template_annotation_json_path, self.dir_result_small)


    
	

	def run(self, user_zip_file, template_name):
		# self.clear_folder(self.dir_user_upload)





		self.extract_file(user_zip_file)

		user_file_name = os.path.basename(user_zip_file).split(".")[0]


		#get the images

		image_files = glob.glob(os.path.join("upload_forms", user_file_name)+ "/*")

		image_files = natsorted(image_files)

		if template_name == "screening_proforma":
			for i, image_file in enumerate(image_files):

				template_file_path = f"form_template_info/screening_proforma/Asha_1_page{i+1}/Asha_1_page{i+1}.png"
				print(template_file_path)

				self.start_engine(template_file_path, image_file)
			final_output = self.merge_final_output(user_file_name, template_name, self.dir_result)

			# print(final_output)

			# self.clear_folder(self.dir_result)


			#final output




		elif template_name == "cbac":

			for i , image_file in enumerate(image_files):
				template_file_path = f"form_template_info/cbac/Cbac_1_page{i+1}/Cbac_1_page{i+1}.png"

				self.start_engine(template_file_path, image_file)

			final_output = self.merge_final_output(user_file_name, template_name, self.dir_result)

			# self.clear_folder(self.dir_result)

			# print(final_output)
		self.clear_folder(self.dir_user_upload)

		return final_output



# parser = argparse.ArgumentParser(description='Asha Digitization_form')
# parser.add_argument('--file_path', metavar='file_path', type=str,
#                     help='Path to the user uploaded image can be zip or tar ')
# parser.add_argument('--template_name', metavar='template_name', type=str, default=None,
#                     help='Please give the perticular template')

# args = parser.parse_args()


# if __name__ == '__main__':



# 	asha_digitization = asha_digitized()


# 	asha_digitization.run("/home/shaon/User_1.zip", "screening_proforma")

