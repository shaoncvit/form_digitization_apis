import easyocr
import torch
import os
import glob
import cv2
import time
import pytesseract
from PIL import Image
import json
from natsort import natsorted
import requests
import shutil


#In this step we don't have to align the images, already aligned
#Just read the images detect the bounding boxs using the json file and then pass it to the ocr
#All are in English so language will be fixed just need to change the options for the ocr models

class english_scanned_digitized:

	def __init__(self):


		self.temp_folder = "temp"
		
		self.form_template_info = "form_template_info"

		self.dir_upload_forms = "upload_forms"
		self.dir_detected_bbox = "detected_bbox"


		self.extra_folder = "extra"


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



	def detect_regions(self, input_image_path, template_name,isPlot = True, count = 1):
		#read the json file for the coordinates and plot if needded and extract the regions
		#until now I am doing it for only one page

		print(f"Input image name:{input_image_path}")

		template_file_name = os.path.join(self.form_template_info, template_name)
		template_page_name = f"page_{count}/page_{count}.json"
		template_annotation_filename = os.path.join(template_file_name, template_page_name)

		print(f"template annotation file path: {template_annotation_filename}")


		#loading the json data
		with open(template_annotation_filename, "r") as file:
			annotation_data = json.load(file)

		#reading the image
		input_image = cv2.imread(input_image_path)
		input_image_plotted = input_image.copy()




		if isPlot:
			for key, items in annotation_data.items():
				for item in items:
					bbox = item["bbox"]

					x,y,w,h = bbox["bbox_x"], bbox["bbox_y"], bbox["bbox_width"], bbox["bbox_height"]

					cv2.rectangle(input_image_plotted, (x,y),(x+w, y+h), (0,0,255),5)


		input_image_name = os.path.basename(input_image_path).split(".")[0]

		input_image_with_plotted_bbox = input_image_name +"_bbox.png"

		cv2.imwrite(os.path.join(self.dir_detected_bbox, input_image_with_plotted_bbox), input_image_plotted)


		#Now need to extract those handwritten regions

		self.clear_folder(self.temp_folder)

		for key, items in annotation_data.items():
			for item in items:
				bbox = item["bbox"]

				x,y,w,h = bbox["bbox_x"], bbox["bbox_y"], bbox["bbox_width"], bbox["bbox_height"]

				if item["text"] == "user":
					region = input_image[y:y+h, x:x+w]

					region_path = os.path.join(self.temp_folder, f"{key}.png")

					cv2.imwrite(region_path, region)




	def call_ocr(self, folder_path, ocr_name):
		print(ocr_name)

		image_files_ocr = natsorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
		# print(image_files_ocr)

		cropped_file_list = [f'temp/{filename}' for filename in image_files_ocr]

		english = []

		if ocr_name == "easyocr":
			#Initialize the EasyOcr reader

			reader = easyocr.Reader(["en"])

			for image_path in cropped_file_list:
				result = reader.readtext(image_path)

				merge_text = ""

				for i in range(len(result)):
					info_ocr = result[i]

					text = info_ocr[1]

					merge_text+= text + " "



				# print(merge_text)

				english.append(merge_text)

			print(english)


		elif ocr_name == "tesseract":

			# pytesseract.pytesseract.tesseract_cmd = (r'/home/shaon/miniconda3/envs/form_ocr/bin/pytesseract')
			# custom_config = r"--lang eng"

			tesseract_url = "https://ilocr.iiit.ac.in/ocr/tesseract"
			payload = {"language": "english"}
			

			headers = {}

			


			for image_path in cropped_file_list:
				# img = cv2.imread(image_path)
				# print(img.shape)

				# text = pytesseract.image_to_string(Image.open(image_path))

				files = [('image', ( '<image-name>.jpg', open(image_path, 'rb'), 'image/jpeg' ) )]
				response = requests.post(tesseract_url, headers=headers, data=payload, files=files) 


				# print(text)

				# english.append(text)
				output = response.json()

				# english.append(output)

				# print(output)

				english.append(output["text"])

			print(english)

		return english


	def mergeOutput(self, template_name, digitized_list, count = 1):

		template_file_name = os.path.join(self.form_template_info, template_name)
		template_page_name = f"page_{count}/page_{count}.json"
		template_annotation_filename = os.path.join(template_file_name, template_page_name)

		template_page_sub_name = os.path.basename(template_page_name).split(".")[0]

		result_json_sub_path = template_page_sub_name + "_output.json"

		os.makedirs("result", exist_ok = True)

		result_json_path = os.path.join("result", result_json_sub_path)


		#load main json file

		with open(template_annotation_filename, "r", encoding = "utf-8") as file:
			main_json = json.load(file)


		string_count = 0
		for key in main_json:
			if isinstance(main_json[key], list):
				for item in main_json[key]:
					if item["text"] == "user" and string_count < len(digitized_list):
						item["text"] = digitized_list[string_count]

						string_count += 1


		#save the updated json to a new file

		with open(result_json_path, "w", encoding= "utf-8") as file:
			json.dump(main_json, file, ensure_ascii = False, indent = 4)


		print(f"updated JSON saved to {result_json_path}")

		return main_json


	def run(self, input_image_path, template_name, ocr_name):
		self.clear_folder(self.temp_folder)

		self.clear_folder(self.extra_folder)
		self.clear_folder(self.dir_detected_bbox)

		self.detect_regions(input_image_path, template_name)

		digitized_texts = self.call_ocr(self.temp_folder,ocr_name)
		# print(digitized_texts)

		final_json = self.mergeOutput(template_name, digitized_texts)

		self.clear_folder(self.dir_upload_forms)

		return final_json
































