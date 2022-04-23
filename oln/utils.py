import cv2
import numpy as np

def get_peak_color(image, bbox):

	region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape((-1,3)).astype(np.float32)
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	compactness,labels,centers = cv2.kmeans(region,1,None,criteria,20,flags)
	return centers[0].astype(np.int32)

def utils_draw_detections(image, detections, scores, draw_scores, mask_alpha):

	mask_img = image.copy()
	det_img = image.copy()
	img_height, img_width = image.shape[:2]

	if draw_scores:
		size = min([img_height, img_width])*0.0008
		text_thickness = int(min([img_height, img_width])*0.0015)

	for bbox, score in zip(detections, scores):

		# Get the common average color using K-Means
		color = get_peak_color(image, bbox)
		max_channel = np.argmax(color)
		color[max_channel] *= 1.5
		color = (int(color[0]),int(color[1]),int(color[2]))
		

		# Draw rectangle
		cv2.rectangle(det_img, (bbox[0], bbox[1]), 
					  (bbox[2], bbox[3]), color, 2)

		# Draw fill rectangle in mask image
		cv2.rectangle(mask_img, (bbox[0], bbox[1]), 
			          (bbox[2], bbox[3]), color, -1)

		if not draw_scores:
			continue

		text = f'{int(score*100)}%'
		(tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
									fontScale=size, thickness=text_thickness)
		
		th = int(th *1.2)
		cv2.rectangle(det_img, (bbox[0], bbox[1]), 
			          (bbox[0]+tw, bbox[1]-th), color, -1)
		cv2.rectangle(mask_img, (bbox[0], bbox[1]), 
			          (bbox[0]+tw, bbox[1]-th), color, -1)
		cv2.putText(det_img, text, (bbox[0], bbox[1]), 
					cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255), text_thickness, cv2.LINE_AA)

		cv2.putText(mask_img, text, (bbox[0], bbox[1]), 
					cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255), text_thickness, cv2.LINE_AA)

	return 	cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)