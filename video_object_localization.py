import cv2
import pafy

from oln import ObjectLocalizationNet, remove_initializer_from_input

# Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'https://youtu.be/vgJUXvkdS78'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 0 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize object localizer
model_path = "models/oln_480x640.onnx"
remove_initializer_from_input(model_path, model_path) # Remove unused nodes
localizer = ObjectLocalizationNet(model_path, threshold=0.75)

cv2.namedWindow("Objects", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue
	
	# Update object localizer
	detections, scores = localizer(frame)

	combined_img = localizer.draw_detections(frame)

	cv2.imshow("Objects", combined_img)