import cv2
import numpy as np
import onnxruntime

from .utils import utils_draw_detections

class ObjectLocalizationNet():

	def __init__(self, model_path, threshold=0.7):

		# Initialize model
		self.initialize_model(model_path, threshold)

	def __call__(self, image):
		return self.detect_objects(image)

	def initialize_model(self, model_path, threshold=0.7):

		self.threshold = threshold

		self.session = onnxruntime.InferenceSession(model_path, 
													providers=['CUDAExecutionProvider', 
															   'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def detect_objects(self, image):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		self.detections, self.scores = self.process_output(outputs)

		return self.detections, self.scores

	def prepare_input(self, image):

		self.img_height, self.img_width = image.shape[:2]

		input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Resize input image
		input_img = cv2.resize(input_img, (self.input_width,self.input_height))  

		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		input_img = ((input_img/ 255.0 - mean) / std)
		input_img = input_img.transpose(2, 0, 1)
		input_tensor = input_img[np.newaxis,:,:,:].astype(np.float32)   

		return input_tensor

	def inference(self, input_tensor):

		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

		return outputs

	def process_output(self, outputs): 

		scores = outputs[0][:,-1]
		bbox = outputs[0][scores>self.threshold,:-1]
		scores = scores[scores>self.threshold]
		
		bbox[:,[1,3]] *= self.img_height/self.input_height
		bbox[:,[0,2]] *= self.img_width/self.input_width

		return bbox.astype(int), scores

	def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

		return utils_draw_detections(image, self.detections, self.scores, 
			                         draw_scores, mask_alpha)

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

if __name__ == '__main__':

	from imread_from_url import imread_from_url
	from remove_initializer_from_input import remove_initializer_from_input

	model_path = "../models/oln_720x1280.onnx"

	# Initialize object localizer
	remove_initializer_from_input(model_path, model_path) # Remove unused nodes
	localizer = ObjectLocalizationNet(model_path, threshold=0.75)

	img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Colorful_plastic_cups_and_mugs.jpg/2560px-Colorful_plastic_cups_and_mugs.jpg")

	# Update object localizer
	detections, scores = localizer(img)

	combined_img = localizer.draw_detections(img)
	cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
	cv2.imshow("Output", combined_img)
	cv2.waitKey(0)