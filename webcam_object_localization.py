import cv2
import pafy

from oln import ObjectLocalizationNet, remove_initializer_from_input

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize object localizer
model_path = "models/oln_360x480.onnx"
remove_initializer_from_input(model_path, model_path) # Remove unused nodes
localizer = ObjectLocalizationNet(model_path, threshold=0.75)

cv2.namedWindow("Objects", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Read frame from the video
	ret, frame = cap.read()

	if not ret:	
		break
	
	# Update object localizer
	detections, scores = localizer(frame)

	combined_img = localizer.draw_detections(frame)
	cv2.imshow("Objects", combined_img)

	# Press key q to stop
	if cv2.waitKey(1) & 0xFF  == ord('q'):
		break
