# from ultralytics import YOLO
# import cv2
#
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv
#
#
# results = {}
#
# mot_tracker = Sort()
#
# # load models
# coco_model = YOLO('yolo11n.pt')
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
# # load video
# cap = cv2.VideoCapture('./Sample3.mp4')
#
# vehicles = [2, 3, 5, 7]
#
# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#         # detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])
#
#         # track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))
#
#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate
#
#             # assign license plate to car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#
#             if car_id != -1:
#
#                 # crop license plate
#                 license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
#
#                 # process license plate
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                 _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
#
#
#                 # read license plate number
#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#                 if license_plate_text is not None:
#                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                     'text': license_plate_text,
#                                                                     'bbox_score': score,
#                                                                     'text_score': license_plate_text_score}}
#
# # write results
# write_csv(results, './test.csv')
# from ultralytics import YOLO
# import cv2
#
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv
#
# results = {}
#
# mot_tracker = Sort()
#
# # load models
# coco_model = YOLO('yolo11n.pt')
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
# # load video
# cap = cv2.VideoCapture('./Sample4.mp4')
#
# vehicles = [2, 3, 5, 7]
#
# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#         # detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])
#
#         # track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))
#
#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate
#
#             # crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
#
#             # process license plate
#             license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#             _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
#
#             # read license plate number
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#             if license_plate_text is not None:
#                 # assign license plate to car (if possible)
#                 xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#
#                 if car_id != -1:
#                     # License plate with associated vehicle
#                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                     'text': license_plate_text,
#                                                                     'bbox_score': score,
#                                                                     'text_score': license_plate_text_score}}
#                 else:
#                     # Standalone license plate (no vehicle detected)
#                     standalone_id = f"LP_{frame_nmr}_{int(x1)}_{int(y1)}"
#                     results[frame_nmr][standalone_id] = {'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                            'text': license_plate_text,
#                                                                            'bbox_score': score,
#                                                                            'text_score': license_plate_text_score}}
#
# # write results
# write_csv(results, './test.csv')
# from ultralytics import YOLO
# import cv2
# import numpy as np
#
# from util import read_license_plate, write_csv
#
# results = {}
#
# # load license plate detection model only
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
# # load video
# cap = cv2.VideoCapture('./Sample2.mp4')
#
# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#
#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate
#
#             # crop license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
#
#             # process license plate
#             license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#             _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
#
#             # read license plate number
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#             if license_plate_text is not None:
#                 # Create unique ID for each license plate detection
#                 plate_id = f"LP_{frame_nmr}_{int(x1)}_{int(y1)}"
#                 results[frame_nmr][plate_id] = {'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                   'text': license_plate_text,
#                                                                   'bbox_score': score,
#                                                                   'text_score': license_plate_text_score}}
#
# # write results
# write_csv(results, './test.csv')

# Correct Version
# from ultralytics import YOLO
# import cv2
#
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv
#
#
# results = {}
#
# mot_tracker = Sort()
#
# # load models
# coco_model = YOLO('yolo11n.pt')
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
# # load video
# cap = cv2.VideoCapture('./IndianSample.mp4')
#
# vehicles = [2, 3, 5, 7]
#
# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#         # detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])
#
#         # track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))
#
#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate
#
#             # assign license plate to car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#
#             if car_id != -1:
#
#                 # crop license plate
#                 license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
#
#                 # process license plate
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                 _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
#
#
#                 # read license plate number
#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#                 if license_plate_text is not None:
#                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                     'text': license_plate_text,
#                                                                     'bbox_score': score,
#                                                                     'text_score': license_plate_text_score}}
#
# # write results
# write_csv(results, './test.csv')


# Tesseract OCR Version (Works)
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv
#
# results = {}
# mot_tracker = Sort()
#
# # Load models
# coco_model = YOLO('yolo11n.pt')
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
# # Load video
# cap = cv2.VideoCapture('Sample3.mp4')
#
#
# vehicles = [2, 3, 5, 7]  # Class IDs for vehicles
#
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#
#         # Detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])
#
#         # Track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))
#
#         # Detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate
#
#             # Add a minimum confidence threshold for license plate detection
#             if score < 0.3:
#                 continue
#
#             # Assign license plate to a car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#             if car_id != -1:
#                 # Add some padding to the license plate crop
#                 padding = 5
#                 y1_crop = max(0, int(y1) - padding)
#                 y2_crop = min(frame.shape[0], int(y2) + padding)
#                 x1_crop = max(0, int(x1) - padding)
#                 x2_crop = min(frame.shape[1], int(x2) + padding)
#
#                 # Crop license plate
#                 license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
#
#                 # Convert to grayscale and threshold
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#
#                 # Try multiple threshold values to improve OCR
#                 threshold_values = [64, 128, 180]
#                 best_text = None
#                 best_score = 0
#
#                 for thresh_val in threshold_values:
#                     _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, thresh_val, 255,
#                                                                  cv2.THRESH_BINARY_INV)
#
#                     # Read license plate
#                     license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#                     if license_plate_text is not None and license_plate_text_score > best_score:
#                         best_text = license_plate_text
#                         best_score = license_plate_text_score
#
#                 if best_text is not None:
#                     # Final validation - ensure it matches Indian license plate format
#                     clean_text = best_text.replace(' ', '').upper()
#                     if len(clean_text) == 10:
#                         # Validate format XX00XX0000
#                         is_valid_format = (
#                                 clean_text[0].isalpha() and clean_text[1].isalpha() and  # XX
#                                 clean_text[2].isdigit() and clean_text[3].isdigit() and  # 00
#                                 clean_text[4].isalpha() and clean_text[5].isalpha() and  # XX
#                                 clean_text[6].isdigit() and clean_text[7].isdigit() and  # 00
#                                 clean_text[8].isdigit() and clean_text[9].isdigit()  # 00
#                         )
#
#                         if is_valid_format:
#                             results[frame_nmr][car_id] = {
#                                 'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                 'license_plate': {
#                                     'bbox': [x1, y1, x2, y2],
#                                     'text': best_text,
#                                     'bbox_score': score,
#                                     'text_score': best_score
#                                 }
#                             }
#                             print(
#                                 f"Frame {frame_nmr}, Car {car_id}: Valid license plate '{best_text}' (format: XX00XX0000) with score {best_score}")
#                         else:
#                             print(
#                                 f"Frame {frame_nmr}, Car {car_id}: Invalid format '{best_text}' - must be XX00XX0000 format - rejected")
#                     else:
#                         print(
#                             f"Frame {frame_nmr}, Car {car_id}: Wrong length '{best_text}' (length: {len(clean_text)}) - rejected")
#
# # Write results
# write_csv(results, './test.csv')
# print(f"Processing complete. Results written to test.csv")
# print(f"Total frames processed: {frame_nmr + 1}")
# print(f"Frames with license plate detections: {len([f for f in results if results[f]])}")
#
# cap.release()

# # Enhanced License Plate Detection with Perspective Transformation and Improved Preprocessing
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from sort.sort import *
# from image_util import get_car, read_license_plate, write_csv
#
#
# def detect_corners_and_apply_perspective_transform(license_plate_crop):
#     """
#     Detect corners of the license plate and apply perspective transformation
#     to get a rectangular, front-facing view of the license plate.
#     """
#     try:
#         # Convert to grayscale for corner detection
#         gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#         # Apply edge detection
#         edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
#
#         # Find contours
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # Find the largest rectangular contour (likely the license plate boundary)
#         largest_contour = None
#         max_area = 0
#
#         for contour in contours:
#             # Approximate the contour
#             epsilon = 0.02 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)
#
#             # Check if the contour has 4 points (rectangular)
#             if len(approx) == 4:
#                 area = cv2.contourArea(contour)
#                 if area > max_area:
#                     max_area = area
#                     largest_contour = approx
#
#         # If we found a rectangular contour, apply perspective transformation
#         if largest_contour is not None and max_area > 100:  # Minimum area threshold
#             # Order the points: top-left, top-right, bottom-right, bottom-left
#             points = largest_contour.reshape(4, 2)
#             ordered_points = order_points(points)
#
#             # Define the dimensions of the output rectangle
#             # Indian license plates have a standard ratio of approximately 4.5:1
#             width = 450  # Width of the output image
#             height = 100  # Height of the output image
#
#             # Define destination points for the perspective transformation
#             dst_points = np.array([
#                 [0, 0],  # Top-left
#                 [width, 0],  # Top-right
#                 [width, height],  # Bottom-right
#                 [0, height]  # Bottom-left
#             ], dtype=np.float32)
#
#             # Calculate the perspective transformation matrix
#             matrix = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst_points)
#
#             # Apply the perspective transformation
#             transformed = cv2.warpPerspective(license_plate_crop, matrix, (width, height))
#
#             return transformed, True
#
#         # If no good rectangular contour found, return original image
#         return license_plate_crop, False
#
#     except Exception as e:
#         print(f"Error in perspective transformation: {e}")
#         return license_plate_crop, False
#
#
# def order_points(pts):
#     """
#     Order points in the order: top-left, top-right, bottom-right, bottom-left
#     """
#     # Initialize the ordered points
#     rect = np.zeros((4, 2), dtype=np.float32)
#
#     # Sum and difference of coordinates
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1)
#
#     # Top-left point has the smallest sum
#     rect[0] = pts[np.argmin(s)]
#
#     # Bottom-right point has the largest sum
#     rect[2] = pts[np.argmax(s)]
#
#     # Top-right point has the smallest difference
#     rect[1] = pts[np.argmin(diff)]
#
#     # Bottom-left point has the largest difference
#     rect[3] = pts[np.argmax(diff)]
#
#     return rect
#
#
# def enhance_license_plate_image(license_plate_crop):
#     """
#     Apply comprehensive image enhancement for better OCR results
#     """
#     try:
#         # Apply perspective transformation first
#         transformed_img, transform_applied = detect_corners_and_apply_perspective_transform(license_plate_crop)
#
#         # Convert to grayscale
#         if len(transformed_img.shape) == 3:
#             gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = transformed_img.copy()
#
#         # Resize if too small (minimum height of 60 pixels for better OCR)
#         height, width = gray.shape
#         if height < 60:
#             scale_factor = 60 / height
#             new_width = int(width * scale_factor)
#             gray = cv2.resize(gray, (new_width, 60), interpolation=cv2.INTER_CUBIC)
#
#         # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
#
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
#
#         # Apply morphological operations to clean up the image
#         # Create kernel for morphological operations
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#
#         # Apply opening (erosion followed by dilation) to remove noise
#         cleaned = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
#
#         # Apply closing (dilation followed by erosion) to close gaps in characters
#         cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
#
#         # Apply unsharp masking for better character definition
#         gaussian = cv2.GaussianBlur(cleaned, (0, 0), 2.0)
#         unsharp_mask = cv2.addWeighted(cleaned, 1.5, gaussian, -0.5, 0)
#
#         return unsharp_mask, transform_applied
#
#     except Exception as e:
#         print(f"Error in image enhancement: {e}")
#         # Return grayscale version as fallback
#         if len(license_plate_crop.shape) == 3:
#             return cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY), False
#         return license_plate_crop, False
#
#
# # Initialize tracking and models
# results = {}
# mot_tracker = Sort()
#
# # Load models
# coco_model = YOLO('yolo11n.pt')
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
# # Load video
# cap = cv2.VideoCapture('/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/frame-190.jpg')
#
# vehicles = [2, 3, 5, 7]  # Class IDs for vehicles
#
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#
#         # Detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])
#
#         # Track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))
#
#         # Detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate
#
#             # Add a minimum confidence threshold for license plate detection
#             if score < 0.3:
#                 continue
#
#             # Assign license plate to a car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#             if car_id != -1:
#                 # Add padding to the license plate crop
#                 padding = 10  # Increased padding for better perspective detection
#                 y1_crop = max(0, int(y1) - padding)
#                 y2_crop = min(frame.shape[0], int(y2) + padding)
#                 x1_crop = max(0, int(x1) - padding)
#                 x2_crop = min(frame.shape[1], int(x2) + padding)
#
#                 # Crop license plate
#                 license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
#
#                 # Apply enhanced preprocessing with perspective transformation
#                 enhanced_license_plate, transform_applied = enhance_license_plate_image(license_plate_crop)
#
#                 # Try multiple threshold values for better OCR
#                 threshold_values = [0, 64, 128, 180]  # Added adaptive threshold (0)
#                 best_text = None
#                 best_score = 0
#
#                 # Also try the enhanced image without additional thresholding
#                 license_plate_text, license_plate_text_score = read_license_plate(enhanced_license_plate)
#                 if license_plate_text is not None and license_plate_text_score > best_score:
#                     best_text = license_plate_text
#                     best_score = license_plate_text_score
#
#                 # Try with different threshold values
#                 for thresh_val in threshold_values:
#                     if thresh_val == 0:
#                         # Adaptive threshold
#                         license_plate_thresh = cv2.adaptiveThreshold(
#                             enhanced_license_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                             cv2.THRESH_BINARY, 11, 2
#                         )
#                     else:
#                         # Fixed threshold
#                         _, license_plate_thresh = cv2.threshold(
#                             enhanced_license_plate, thresh_val, 255, cv2.THRESH_BINARY
#                         )
#
#                     # Also try inverted threshold
#                     license_plate_thresh_inv = cv2.bitwise_not(license_plate_thresh)
#
#                     # Test both regular and inverted threshold
#                     for thresh_img in [license_plate_thresh, license_plate_thresh_inv]:
#                         license_plate_text, license_plate_text_score = read_license_plate(thresh_img)
#                         if license_plate_text is not None and license_plate_text_score > best_score:
#                             best_text = license_plate_text
#                             best_score = license_plate_text_score
#
#                 if best_text is not None:
#                     # Final validation - ensure it matches Indian license plate format
#                     clean_text = best_text.replace(' ', '').upper()
#                     if len(clean_text) == 10:
#                         # Validate format XX00XX0000
#                         is_valid_format = (
#                                 clean_text[0].isalpha() and clean_text[1].isalpha() and  # XX
#                                 clean_text[2].isdigit() and clean_text[3].isdigit() and  # 00
#                                 clean_text[4].isalpha() and clean_text[5].isalpha() and  # XX
#                                 clean_text[6].isdigit() and clean_text[7].isdigit() and  # 00
#                                 clean_text[8].isdigit() and clean_text[9].isdigit()  # 00
#                         )
#
#                         if is_valid_format:
#                             results[frame_nmr][car_id] = {
#                                 'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                 'license_plate': {
#                                     'bbox': [x1, y1, x2, y2],
#                                     'text': best_text,
#                                     'bbox_score': score,
#                                     'text_score': best_score,
#                                     'perspective_transform_applied': transform_applied
#                                 }
#                             }
#                             transform_msg = "with perspective correction" if transform_applied else "without perspective correction"
#                             print(f"Frame {frame_nmr}, Car {car_id}: Valid license plate '{best_text}' "
#                                   f"(format: XX00XX0000) with score {best_score} {transform_msg}")
#                         else:
#                             print(f"Frame {frame_nmr}, Car {car_id}: Invalid format '{best_text}' "
#                                   f"- must be XX00XX0000 format - rejected")
#                     else:
#                         print(f"Frame {frame_nmr}, Car {car_id}: Wrong length '{best_text}' "
#                               f"(length: {len(clean_text)}) - rejected")
#
# # Write results
# write_csv(results, './test.csv')
# print(f"Processing complete. Results written to test.csv")
# print(f"Total frames processed: {frame_nmr + 1}")
# print(f"Frames with license plate detections: {len([f for f in results if results[f]])}")
#
# # Print statistics about perspective transformation usage
# perspective_count = sum(1 for frame in results.values()
#                         for car in frame.values()
#                         if car.get('license_plate', {}).get('perspective_transform_applied', False))
# total_detections = sum(len(frame) for frame in results.values())
# print(f"Perspective transformation applied: {perspective_count}/{total_detections} detections")
#
# cap.release()


# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# from sort.sort import *
# from image_util import get_car, read_license_plate, write_csv
# # Import your perspective transformation functions
# from image_main import detect_corners_and_apply_perspective_transform, enhance_license_plate_image
#
#
# def process_image_core(frame, coco_model, license_plate_detector):
#     """
#     Core image processing logic that can be used for both single and batch processing
#     """
#     vehicles = [2, 3, 5, 7]  # Class IDs for vehicles
#     image_results = {}
#
#     # Detect vehicles
#     detections = coco_model(frame)[0]
#     detections_ = []
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             detections_.append([x1, y1, x2, y2, score])
#
#     # For images, assign simple IDs to vehicles (no tracking needed)
#     track_ids = []
#     for i, detection in enumerate(detections_):
#         x1, y1, x2, y2, score = detection
#         track_ids.append([x1, y1, x2, y2, i])  # Use index as ID
#
#     # Detect license plates
#     license_plates = license_plate_detector(frame)[0]
#     for license_plate in license_plates.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = license_plate
#
#         # Add a minimum confidence threshold for license plate detection
#         if score < 0.3:
#             continue
#
#         # Assign license plate to a car
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#         if car_id != -1:
#             # Add padding to the license plate crop
#             padding = 10
#             y1_crop = max(0, int(y1) - padding)
#             y2_crop = min(frame.shape[0], int(y2) + padding)
#             x1_crop = max(0, int(x1) - padding)
#             x2_crop = min(frame.shape[1], int(x2) + padding)
#
#             # Crop license plate
#             license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
#
#             # Apply enhanced preprocessing with perspective transformation
#             enhanced_license_plate, transform_applied = enhance_license_plate_image(license_plate_crop)
#
#             # Try multiple threshold values for better OCR
#             threshold_values = [0, 64, 128, 180]
#             best_text = None
#             best_score = 0
#
#             # Also try the enhanced image without additional thresholding
#             license_plate_text, license_plate_text_score = read_license_plate(enhanced_license_plate)
#             if license_plate_text is not None and license_plate_text_score > best_score:
#                 best_text = license_plate_text
#                 best_score = license_plate_text_score
#
#             # Try with different threshold values
#             for thresh_val in threshold_values:
#                 if thresh_val == 0:
#                     # Adaptive threshold
#                     license_plate_thresh = cv2.adaptiveThreshold(
#                         enhanced_license_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                         cv2.THRESH_BINARY, 11, 2
#                     )
#                 else:
#                     # Fixed threshold
#                     _, license_plate_thresh = cv2.threshold(
#                         enhanced_license_plate, thresh_val, 255, cv2.THRESH_BINARY
#                     )
#
#                 # Also try inverted threshold
#                 license_plate_thresh_inv = cv2.bitwise_not(license_plate_thresh)
#
#                 # Test both regular and inverted threshold
#                 for thresh_img in [license_plate_thresh, license_plate_thresh_inv]:
#                     license_plate_text, license_plate_text_score = read_license_plate(thresh_img)
#                     if license_plate_text is not None and license_plate_text_score > best_score:
#                         best_text = license_plate_text
#                         best_score = license_plate_text_score
#
#             if best_text is not None:
#                 # Final validation - ensure it matches Indian license plate format
#                 clean_text = best_text.replace(' ', '').upper()
#                 if len(clean_text) == 10:
#                     # Validate format XX00XX0000
#                     is_valid_format = (
#                             clean_text[0].isalpha() and clean_text[1].isalpha() and  # XX
#                             clean_text[2].isdigit() and clean_text[3].isdigit() and  # 00
#                             clean_text[4].isalpha() and clean_text[5].isalpha() and  # XX
#                             clean_text[6].isdigit() and clean_text[7].isdigit() and  # 00
#                             clean_text[8].isdigit() and clean_text[9].isdigit()  # 00
#                     )
#
#                     if is_valid_format:
#                         image_results[car_id] = {
#                             'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                             'license_plate': {
#                                 'bbox': [x1, y1, x2, y2],
#                                 'text': best_text,
#                                 'bbox_score': score,
#                                 'text_score': best_score,
#                                 'perspective_transform_applied': transform_applied
#                             }
#                         }
#                         transform_msg = "with perspective correction" if transform_applied else "without perspective correction"
#                         print(f"Car {car_id}: Valid license plate '{best_text}' "
#                               f"(format: XX00XX0000) with score {best_score} {transform_msg}")
#
#     return image_results
#
#
# def process_single_image(image_path, output_csv_path):
#     """
#     Process a single image for license plate detection and recognition
#     """
#     # Load models
#     coco_model = YOLO('yolo11n.pt')
#     license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
#     # Load image
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"Error: Could not load image from {image_path}")
#         return None
#
#     print(f"Processing single image: {image_path}")
#
#     # Process the image
#     image_results = process_image_core(frame, coco_model, license_plate_detector)
#
#     # Format results for CSV (single image = frame 0)
#     results = {0: image_results}
#
#     # Write results
#     write_csv(results, output_csv_path)
#     print(f"Processing complete. Results written to {output_csv_path}")
#
#     return results
#
#
# def process_multiple_images(image_paths, output_csv_path):
#     """
#     Process multiple images and combine results
#     """
#     # Load models once for all images
#     coco_model = YOLO('yolo11n.pt')
#     license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')
#
#     combined_results = {}
#
#     for frame_idx, image_path in enumerate(image_paths):
#         print(f"Processing image {frame_idx + 1}/{len(image_paths)}: {image_path}")
#
#         # Load image
#         frame = cv2.imread(image_path)
#         if frame is None:
#             print(f"Error: Could not load image from {image_path}")
#             continue
#
#         # Process the image
#         image_results = process_image_core(frame, coco_model, license_plate_detector)
#
#         # Add to combined results with frame index
#         if image_results:  # Only add if we found license plates
#             combined_results[frame_idx] = image_results
#
#     # Write combined results
#     write_csv(combined_results, output_csv_path)
#     print(f"Batch processing complete. Processed {len(image_paths)} images.")
#     print(f"Found license plates in {len(combined_results)} images.")
#     print(f"Results written to {output_csv_path}")
#
#     return combined_results
#
#
# def process_image_folder(folder_path, output_csv_path, image_extensions=None):
#     """
#     Process all images in a folder
#     """
#     if image_extensions is None:
#         image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
#
#     # Get all image files from folder
#     image_paths = []
#     for filename in os.listdir(folder_path):
#         if any(filename.lower().endswith(ext) for ext in image_extensions):
#             image_paths.append(os.path.join(folder_path, filename))
#
#     if not image_paths:
#         print(f"No image files found in {folder_path}")
#         return None
#
#     print(f"Found {len(image_paths)} images in {folder_path}")
#
#     # Process all images
#     return process_multiple_images(image_paths, output_csv_path)
#
#
# # Example usage
# if __name__ == "__main__":
#     # Option 1: Process single image
#     single_image_path = "/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/frame-190.jpg"
#     process_single_image(single_image_path, "single_image_results.csv")
#
#     # Option 2: Process specific list of images
#     image_list = ["image1.jpg", "image2.jpg", "image3.jpg"]
#     process_multiple_images(image_list, "batch_results.csv")
#
#     # Option 3: Process entire folder
#     folder_path = "path/to/image/folder"
#     process_image_folder(folder_path, "folder_results.csv")


# Enhanced License Plate Detection with Perspective Transformation and Improved Preprocessing
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from image_util import get_car, read_license_plate, write_csv


def detect_corners_and_apply_perspective_transform(license_plate_crop):
    """
    Detect corners of the license plate and apply perspective transformation
    to get a rectangular, front-facing view of the license plate.
    """
    try:
        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour (likely the license plate boundary)
        largest_contour = None
        max_area = 0

        for contour in contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour has 4 points (rectangular)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = approx

        # If we found a rectangular contour, apply perspective transformation
        if largest_contour is not None and max_area > 100:  # Minimum area threshold
            # Order the points: top-left, top-right, bottom-right, bottom-left
            points = largest_contour.reshape(4, 2)
            ordered_points = order_points(points)

            # Define the dimensions of the output rectangle
            # Indian license plates have a standard ratio of approximately 4.5:1
            width = 450  # Width of the output image
            height = 100  # Height of the output image

            # Define destination points for the perspective transformation
            dst_points = np.array([
                [0, 0],  # Top-left
                [width, 0],  # Top-right
                [width, height],  # Bottom-right
                [0, height]  # Bottom-left
            ], dtype=np.float32)

            # Calculate the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst_points)

            # Apply the perspective transformation
            transformed = cv2.warpPerspective(license_plate_crop, matrix, (width, height))

            return transformed, True

        # If no good rectangular contour found, return original image
        return license_plate_crop, False

    except Exception as e:
        print(f"Error in perspective transformation: {e}")
        return license_plate_crop, False


def order_points(pts):
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left
    """
    # Initialize the ordered points
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum and difference of coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Top-left point has the smallest sum
    rect[0] = pts[np.argmin(s)]

    # Bottom-right point has the largest sum
    rect[2] = pts[np.argmax(s)]

    # Top-right point has the smallest difference
    rect[1] = pts[np.argmin(diff)]

    # Bottom-left point has the largest difference
    rect[3] = pts[np.argmax(diff)]

    return rect


def enhance_license_plate_image(license_plate_crop):
    """
    Apply comprehensive image enhancement for better OCR results
    """
    try:
        # Apply perspective transformation first
        transformed_img, transform_applied = detect_corners_and_apply_perspective_transform(license_plate_crop)

        # Convert to grayscale
        if len(transformed_img.shape) == 3:
            gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = transformed_img.copy()

        # Resize if too small (minimum height of 60 pixels for better OCR)
        height, width = gray.shape
        if height < 60:
            scale_factor = 60 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 60), interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Apply morphological operations to clean up the image
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Apply opening (erosion followed by dilation) to remove noise
        cleaned = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

        # Apply closing (dilation followed by erosion) to close gaps in characters
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # Apply unsharp masking for better character definition
        gaussian = cv2.GaussianBlur(cleaned, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(cleaned, 1.5, gaussian, -0.5, 0)

        return unsharp_mask, transform_applied

    except Exception as e:
        print(f"Error in image enhancement: {e}")
        # Return grayscale version as fallback
        if len(license_plate_crop.shape) == 3:
            return cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY), False
        return license_plate_crop, False


def main():
    """
    Main function to run video processing
    """
    # Initialize tracking and models
    results = {}
    mot_tracker = Sort()

    # Load models
    coco_model = YOLO('yolo11n.pt')
    license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')

    # Load video
    cap = cv2.VideoCapture('Sample3.mp4')

    vehicles = [2, 3, 5, 7]  # Class IDs for vehicles

    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}

            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Add a minimum confidence threshold for license plate detection
                if score < 0.3:
                    continue

                # Assign license plate to a car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                if car_id != -1:
                    # Add padding to the license plate crop
                    padding = 10  # Increased padding for better perspective detection
                    y1_crop = max(0, int(y1) - padding)
                    y2_crop = min(frame.shape[0], int(y2) + padding)
                    x1_crop = max(0, int(x1) - padding)
                    x2_crop = min(frame.shape[1], int(x2) + padding)

                    # Crop license plate
                    license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                    # Apply enhanced preprocessing with perspective transformation
                    enhanced_license_plate, transform_applied = enhance_license_plate_image(license_plate_crop)

                    # Try multiple threshold values for better OCR
                    threshold_values = [0, 64, 128, 180]  # Added adaptive threshold (0)
                    best_text = None
                    best_score = 0

                    # Also try the enhanced image without additional thresholding
                    license_plate_text, license_plate_text_score = read_license_plate(enhanced_license_plate)
                    if license_plate_text is not None and license_plate_text_score > best_score:
                        best_text = license_plate_text
                        best_score = license_plate_text_score

                    # Try with different threshold values
                    for thresh_val in threshold_values:
                        if thresh_val == 0:
                            # Adaptive threshold
                            license_plate_thresh = cv2.adaptiveThreshold(
                                enhanced_license_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2
                            )
                        else:
                            # Fixed threshold
                            _, license_plate_thresh = cv2.threshold(
                                enhanced_license_plate, thresh_val, 255, cv2.THRESH_BINARY
                            )

                        # Also try inverted threshold
                        license_plate_thresh_inv = cv2.bitwise_not(license_plate_thresh)

                        # Test both regular and inverted threshold
                        for thresh_img in [license_plate_thresh, license_plate_thresh_inv]:
                            license_plate_text, license_plate_text_score = read_license_plate(thresh_img)
                            if license_plate_text is not None and license_plate_text_score > best_score:
                                best_text = license_plate_text
                                best_score = license_plate_text_score

                    if best_text is not None:
                        # Final validation - ensure it matches Indian license plate format
                        clean_text = best_text.replace(' ', '').upper()
                        if len(clean_text) == 10:
                            # Validate format XX00XX0000
                            is_valid_format = (
                                    clean_text[0].isalpha() and clean_text[1].isalpha() and  # XX
                                    clean_text[2].isdigit() and clean_text[3].isdigit() and  # 00
                                    clean_text[4].isalpha() and clean_text[5].isalpha() and  # XX
                                    clean_text[6].isdigit() and clean_text[7].isdigit() and  # 00
                                    clean_text[8].isdigit() and clean_text[9].isdigit()  # 00
                            )

                            if is_valid_format:
                                results[frame_nmr][car_id] = {
                                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                    'license_plate': {
                                        'bbox': [x1, y1, x2, y2],
                                        'text': best_text,
                                        'bbox_score': score,
                                        'text_score': best_score,
                                        'perspective_transform_applied': transform_applied
                                    }
                                }
                                transform_msg = "with perspective correction" if transform_applied else "without perspective correction"
                                print(f"Frame {frame_nmr}, Car {car_id}: Valid license plate '{best_text}' "
                                      f"(format: XX00XX0000) with score {best_score} {transform_msg}")
                            else:
                                print(f"Frame {frame_nmr}, Car {car_id}: Invalid format '{best_text}' "
                                      f"- must be XX00XX0000 format - rejected")
                        else:
                            print(f"Frame {frame_nmr}, Car {car_id}: Wrong length '{best_text}' "
                                  f"(length: {len(clean_text)}) - rejected")

    # Write results
    write_csv(results, './test.csv')
    print(f"Processing complete. Results written to test.csv")
    print(f"Total frames processed: {frame_nmr + 1}")
    print(f"Frames with license plate detections: {len([f for f in results if results[f]])}")

    # Print statistics about perspective transformation usage
    perspective_count = sum(1 for frame in results.values()
                            for car in frame.values()
                            if car.get('license_plate', {}).get('perspective_transform_applied', False))
    total_detections = sum(len(frame) for frame in results.values())
    print(f"Perspective transformation applied: {perspective_count}/{total_detections} detections")

    cap.release()


# This ensures the main execution only happens when the script is run directly
if __name__ == "__main__":
    main()