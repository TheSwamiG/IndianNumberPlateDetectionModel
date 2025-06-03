# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# import glob
# from image_util import get_car_image, read_license_plate, draw_border, validate_indian_license_plate
#
# # ========================================
# # CONFIGURATION SECTION - MODIFY THESE PATHS
# # ========================================
#
# # 1. INPUT CONFIGURATION
# INPUT_PATH = "/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/NumberPlateImages/NumberPlate6.jpeg"
#
# # 2. OUTPUT CONFIGURATION
# OUTPUT_DIR = "/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/DetectionResults"
#
# # 3. MODEL CONFIGURATION
# COCO_MODEL_PATH = "yolo11n.pt"
# LICENSE_PLATE_MODEL_PATH = "/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt"
#
# # 4. DETECTION SETTINGS
# MIN_CONFIDENCE = 0.1
# CROP_PADDING = 10  # Increased padding for better OCR
#
#
# def process_single_image(image_path, coco_model, license_plate_detector, output_dir):
#     """Process a single image for license plate detection with improved OCR"""
#
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"Error: Could not load image {image_path}")
#         return None
#
#     print(f"Processing image: {os.path.basename(image_path)}")
#
#     height, width = frame.shape[:2]
#     vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
#
#     # Detect vehicles
#     detections = coco_model(frame)[0]
#     vehicle_detections = []
#
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             vehicle_detections.append([x1, y1, x2, y2, score, int(class_id)])
#
#     print(f"Found {len(vehicle_detections)} vehicles")
#
#     # Detect license plates
#     license_plates = license_plate_detector(frame)[0]
#
#     annotated_frame = frame.copy()
#     detections_data = []
#
#     for lp_idx, license_plate in enumerate(license_plates.boxes.data.tolist()):
#         x1, y1, x2, y2, score, class_id = license_plate
#
#         if score < MIN_CONFIDENCE:
#             continue
#
#         print(f"Processing license plate {lp_idx + 1} with confidence {score:.2f}")
#
#         # Find associated vehicle
#         associated_vehicle = get_car_image(license_plate, vehicle_detections)
#
#         # Add padding to license plate crop
#         y1_crop = max(0, int(y1) - CROP_PADDING)
#         y2_crop = min(height, int(y2) + CROP_PADDING)
#         x1_crop = max(0, int(x1) - CROP_PADDING)
#         x2_crop = min(width, int(x2) + CROP_PADDING)
#
#         # Crop license plate
#         license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
#
#         if license_plate_crop.size > 0:
#             # Read license plate using improved OCR
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
#
#             print(f"OCR Result: '{license_plate_text}' with confidence {license_plate_text_score:.2f}")
#
#             # Validate license plate format with more flexibility
#             is_valid, validation_confidence, formatted_text = validate_indian_license_plate(license_plate_text)
#
#             if is_valid and validation_confidence > 30:  # Reduced threshold for more flexibility
#                 print(
#                     f"Valid license plate detected: '{formatted_text}' (validation confidence: {validation_confidence})")
#
#                 # Draw vehicle bounding box if available
#                 if associated_vehicle is not None:
#                     vx1, vy1, vx2, vy2, v_score, v_class = associated_vehicle
#                     draw_border(annotated_frame, (int(vx1), int(vy1)), (int(vx2), int(vy2)),
#                                 (0, 255, 0), 25, line_length_x=200, line_length_y=200)
#
#                     # Position text above vehicle
#                     text_x = int((vx1 + vx2) / 2) - 100
#                     text_y = max(50, int(vy1) - 30)
#                 else:
#                     # Position text above license plate if no vehicle found
#                     text_x = int(x1)
#                     text_y = max(50, int(y1) - 10)
#
#                 # Draw license plate bounding box
#                 cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)),
#                               (0, 0, 255), 12)
#
#                 # Add background rectangle for text
#                 display_text = formatted_text or license_plate_text
#                 text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
#                 cv2.rectangle(annotated_frame,
#                               (text_x - 10, text_y - text_size[1] - 10),
#                               (text_x + text_size[0] + 10, text_y + 10),
#                               (255, 255, 255), -1)
#
#                 # Add license plate text
#                 cv2.putText(annotated_frame, display_text,
#                             (text_x, text_y),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             2, (0, 0, 0), 3)
#
#                 # Save individual crop with OCR result
#                 save_license_plate_crop(license_plate_crop, display_text, output_dir,
#                                         os.path.basename(image_path), lp_idx)
#
#                 # Store detection data
#                 detection_info = {
#                     'vehicle_bbox': associated_vehicle[:4] if associated_vehicle else None,
#                     'license_plate_bbox': [x1, y1, x2, y2],
#                     'license_plate_text': display_text,
#                     'license_plate_confidence': license_plate_text_score,
#                     'validation_confidence': validation_confidence,
#                     'vehicle_confidence': associated_vehicle[4] if associated_vehicle else 0,
#                     'vehicle_type': associated_vehicle[5] if associated_vehicle else None
#                 }
#                 detections_data.append(detection_info)
#             else:
#                 print(
#                     f"License plate rejected: '{license_plate_text}' (validation confidence: {validation_confidence})")
#
#                 # Still save the crop for debugging
#                 debug_text = f"REJECTED_{license_plate_text}" if license_plate_text else "NO_TEXT"
#                 save_license_plate_crop(license_plate_crop, debug_text, output_dir,
#                                         os.path.basename(image_path), f"debug_{lp_idx}")
#
#     # Save annotated image
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
#     cv2.imwrite(output_path, annotated_frame)
#
#     return {
#         'image_path': image_path,
#         'output_path': output_path,
#         'detections': detections_data,
#         'total_detections': len(detections_data)
#     }
#
#
# def save_license_plate_crop(crop_image, text, output_dir, image_name, plate_idx):
#     """Save cropped license plate image with text overlay"""
#     if crop_image is None or crop_image.size == 0:
#         return None
#
#     # Create crops directory
#     crops_dir = os.path.join(output_dir, 'crops')
#     os.makedirs(crops_dir, exist_ok=True)
#
#     # Resize crop for better visibility (minimum height of 100 pixels)
#     height, width = crop_image.shape[:2]
#     if height < 100:
#         scale_factor = 100 / height
#         new_width = int(width * scale_factor)
#         crop_resized = cv2.resize(crop_image, (new_width, 100), interpolation=cv2.INTER_CUBIC)
#     else:
#         crop_resized = crop_image.copy()
#
#     # Add text overlay
#     if text:
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.8
#         font_thickness = 2
#         text_color = (0, 255, 0) if not text.startswith('REJECTED') else (0, 0, 255)
#
#         # Get text size
#         (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
#
#         # Create a larger image to accommodate text
#         new_height = crop_resized.shape[0] + text_height + baseline + 30
#         new_width = max(crop_resized.shape[1], text_width + 20)
#
#         result_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
#         result_img.fill(255)  # White background
#
#         # Place the crop image
#         y_offset = 10
#         x_offset = (new_width - crop_resized.shape[1]) // 2
#         result_img[y_offset:y_offset + crop_resized.shape[0],
#         x_offset:x_offset + crop_resized.shape[1]] = crop_resized
#
#         # Add text below the image
#         text_x = (new_width - text_width) // 2
#         text_y = crop_resized.shape[0] + y_offset + text_height + 15
#         cv2.putText(result_img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
#
#         crop_to_save = result_img
#     else:
#         crop_to_save = crop_resized
#
#     # Save the crop
#     base_name = os.path.splitext(image_name)[0]
#     crop_filename = f"{base_name}_plate_{plate_idx}.jpg"
#     crop_path = os.path.join(crops_dir, crop_filename)
#
#     cv2.imwrite(crop_path, crop_to_save)
#     print(f"Saved crop: {crop_path}")
#     return crop_path
#
#
# def main():
#     """Main function to run license plate detection"""
#
#     print("=" * 60)
#     print("LICENSE PLATE DETECTION SYSTEM")
#     print("=" * 60)
#
#     # Create output directory
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     print(f"Output directory: {OUTPUT_DIR}")
#
#     # Load models
#     print("\nLoading models...")
#     try:
#         print(f"Loading COCO model: {COCO_MODEL_PATH}")
#         coco_model = YOLO(COCO_MODEL_PATH)
#
#         print(f"Loading license plate model: {LICENSE_PLATE_MODEL_PATH}")
#         license_plate_detector = YOLO(LICENSE_PLATE_MODEL_PATH)
#
#         print("✓ Models loaded successfully!")
#     except Exception as e:
#         print(f"✗ Error loading models: {e}")
#         return
#
#     # Get list of images to process
#     if os.path.isfile(INPUT_PATH):
#         image_paths = [INPUT_PATH]
#         print(f"\nProcessing single image: {INPUT_PATH}")
#     elif os.path.isdir(INPUT_PATH):
#         extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
#         image_paths = []
#         for ext in extensions:
#             image_paths.extend(glob.glob(os.path.join(INPUT_PATH, ext)))
#             image_paths.extend(glob.glob(os.path.join(INPUT_PATH, ext.upper())))
#         print(f"\nProcessing directory: {INPUT_PATH}")
#     else:
#         print(f"✗ Error: {INPUT_PATH} is not a valid file or directory")
#         return
#
#     if not image_paths:
#         print(f"✗ No images found in {INPUT_PATH}")
#         return
#
#     print(f"Found {len(image_paths)} images to process")
#     print(f"Using confidence threshold: {MIN_CONFIDENCE}")
#
#     # Process images
#     all_results = []
#     successful_detections = 0
#
#     for i, image_path in enumerate(image_paths, 1):
#         print(f"\n--- Processing image {i}/{len(image_paths)} ---")
#
#         result = process_single_image(image_path, coco_model, license_plate_detector, OUTPUT_DIR)
#
#         if result:
#             all_results.append(result)
#             if result['total_detections'] > 0:
#                 successful_detections += 1
#                 print(f"✓ Saved annotated image: {result['output_path']}")
#                 print(f"  Found {result['total_detections']} valid license plate(s)")
#             else:
#                 print(f"✗ No valid license plates detected")
#         else:
#             print(f"✗ Failed to process image")
#
#     # Print summary
#     print(f"\n{'=' * 50}")
#     print(f"PROCESSING SUMMARY")
#     print(f"{'=' * 50}")
#     print(f"Total images processed: {len(image_paths)}")
#     print(f"Images with license plates detected: {successful_detections}")
#     print(f"Total license plates found: {sum(r['total_detections'] for r in all_results)}")
#     print(f"Output directory: {OUTPUT_DIR}")
#     print(f"Individual crops saved in: {os.path.join(OUTPUT_DIR, 'crops')}")
#
#     # Save summary report
#     summary_path = os.path.join(OUTPUT_DIR, 'detection_summary.txt')
#     with open(summary_path, 'w') as f:
#         f.write("License Plate Detection Summary\n")
#         f.write("=" * 40 + "\n\n")
#         f.write(f"Input: {INPUT_PATH}\n")
#         f.write(f"Output: {OUTPUT_DIR}\n")
#         f.write(f"Confidence Threshold: {MIN_CONFIDENCE}\n\n")
#
#         for result in all_results:
#             f.write(f"Image: {os.path.basename(result['image_path'])}\n")
#             f.write(f"Detections: {result['total_detections']}\n")
#
#             for i, detection in enumerate(result['detections'], 1):
#                 f.write(f"  {i}. License Plate: {detection['license_plate_text']}\n")
#                 f.write(f"     OCR Confidence: {detection['license_plate_confidence']:.2f}\n")
#                 f.write(f"     Validation Confidence: {detection['validation_confidence']:.2f}\n")
#             f.write("\n")
#
#     print(f"Summary saved to: {summary_path}")
#
#
# if __name__ == "__main__":
#     main()
#

#
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from util import get_car, read_license_plate, write_csv
#
# # Load models
# coco_model = YOLO('yolo11n.pt')  # Vehicle detection model
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')  # License plate model
#
# vehicles = [2, 3, 5, 7]  # Vehicle class IDs
#
# def detect_license_plates_in_image(image_path, output_csv='results.csv'):
#     frame_nmr = 0
#     results = {}
#
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"Could not read image {image_path}")
#         return
#
#     results[frame_nmr] = {}
#
#     # Detect vehicles
#     detections = coco_model(frame)[0]
#     detections_ = []
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             detections_.append([x1, y1, x2, y2, score])
#
#     # For single image, no tracking; just assign IDs by index
#     track_ids = []
#     for idx, det in enumerate(detections_):
#         track_ids.append(np.array([det[0], det[1], det[2], det[3], idx]))
#
#     track_ids = np.array(track_ids)
#
#     # Detect license plates
#     license_plates = license_plate_detector(frame)[0]
#
#     for license_plate in license_plates.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = license_plate
#         if score < 0.3:
#             continue
#
#         # Assign license plate to car
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#         if car_id == -1:
#             continue
#
#         padding = 5
#         y1_crop = max(0, int(y1) - padding)
#         y2_crop = min(frame.shape[0], int(y2) + padding)
#         x1_crop = max(0, int(x1) - padding)
#         x2_crop = min(frame.shape[1], int(x2) + padding)
#
#         license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
#         license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#
#         threshold_values = [64, 128, 180]
#         best_text = None
#         best_score = 0
#
#         for thresh_val in threshold_values:
#             _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
#
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#             if license_plate_text is not None and license_plate_text_score > best_score:
#                 best_text = license_plate_text
#                 best_score = license_plate_text_score
#
#         if best_text is not None:
#             clean_text = best_text.replace(' ', '').upper()
#             if len(clean_text) == 10:
#                 # Validate Indian format XX00XX0000
#                 is_valid_format = (
#                     clean_text[0].isalpha() and clean_text[1].isalpha() and
#                     clean_text[2].isdigit() and clean_text[3].isdigit() and
#                     clean_text[4].isalpha() and clean_text[5].isalpha() and
#                     clean_text[6].isdigit() and clean_text[7].isdigit() and
#                     clean_text[8].isdigit() and clean_text[9].isdigit()
#                 )
#
#                 if is_valid_format:
#                     results[frame_nmr][car_id] = {
#                         'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                         'license_plate': {
#                             'bbox': [x1, y1, x2, y2],
#                             'text': best_text,
#                             'bbox_score': score,
#                             'text_score': best_score
#                         }
#                     }
#                     print(f"Car {car_id}: Valid license plate '{best_text}' with score {best_score}")
#                 else:
#                     print(f"Car {car_id}: Invalid format '{best_text}' - rejected")
#             else:
#                 print(f"Car {car_id}: Wrong length '{best_text}' - rejected")
#
#     # Draw bounding boxes and text on the image
#     for car_id, data in results[frame_nmr].items():
#         # Draw car bbox (blue)
#         x1, y1, x2, y2 = map(int, data['car']['bbox'])
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
#         # Draw license plate bbox (green)
#         x1_lp, y1_lp, x2_lp, y2_lp = map(int, data['license_plate']['bbox'])
#         cv2.rectangle(frame, (x1_lp, y1_lp), (x2_lp, y2_lp), (0, 255, 0), 2)
#
#         # Put recognized license plate text above license plate bbox
#         lp_text = data['license_plate']['text']
#         cv2.putText(frame, lp_text, (x1_lp, y1_lp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#     # Save annotated image
#     output_image_path = "output_with_detections.jpg"
#     cv2.imwrite(output_image_path, frame)
#     print(f"Annotated image saved as {output_image_path}")
#
#     # Save CSV results
#     write_csv(results, output_csv)
#     print(f"Detection complete. Results saved to {output_csv}")
#
# if __name__ == '__main__':
#     image_path = '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/NumberPlateImages/NumberPlate3.jpg'  # Replace with your image path
#     detect_license_plates_in_image(image_path)


# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# from sort.sort import *
# from image_util import get_car, read_license_plate, write_csv
#
# # Initialize model
# coco_model = YOLO('yolo11n.pt')  # Vehicle detection model
# license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')  # Number plate model
#
# mot_tracker = Sort()
# vehicles = [2, 3, 5, 7]  # YOLO class IDs for vehicles
#
# # Load single image
# image_path = '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/frame-190.jpg'
# frame = cv2.imread(image_path)
# if frame is None:
#     raise FileNotFoundError(f"Image not found at path: {image_path}")
#
# results = {}
#
# # Detect vehicles
# detections = coco_model(frame)[0]
# detections_ = []
# for detection in detections.boxes.data.tolist():
#     x1, y1, x2, y2, score, class_id = detection
#     if int(class_id) in vehicles:
#         detections_.append([x1, y1, x2, y2, score])
#
# # Track vehicles
# track_ids = mot_tracker.update(np.asarray(detections_))
#
# # Detect license plates
# license_plates = license_plate_detector(frame)[0]
# for license_plate in license_plates.boxes.data.tolist():
#     x1, y1, x2, y2, score, class_id = license_plate
#     if score < 0.3:
#         continue
#
#     # Associate license plate with a vehicle
#     xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#     if car_id != -1:
#         padding = 5
#         y1_crop = max(0, int(y1) - padding)
#         y2_crop = min(frame.shape[0], int(y2) + padding)
#         x1_crop = max(0, int(x1) - padding)
#         x2_crop = min(frame.shape[1], int(x2) + padding)
#
#         license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
#         license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#
#         threshold_values = [64, 128, 180]
#         best_text = None
#         best_score = 0
#
#         for thresh_val in threshold_values:
#             _, thresh = cv2.threshold(license_plate_crop_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
#             license_plate_text, score = read_license_plate(thresh)
#             if license_plate_text and score > best_score:
#                 best_text = license_plate_text
#                 best_score = score
#
#         if best_text:
#             clean_text = best_text.replace(' ', '').upper()
#             if len(clean_text) == 10:
#                 is_valid_format = (
#                     clean_text[0].isalpha() and clean_text[1].isalpha() and
#                     clean_text[2].isdigit() and clean_text[3].isdigit() and
#                     clean_text[4].isalpha() and clean_text[5].isalpha() and
#                     clean_text[6].isdigit() and clean_text[7].isdigit() and
#                     clean_text[8].isdigit() and clean_text[9].isdigit()
#                 )
#
#                 if is_valid_format:
#                     results[0] = {
#                         car_id: {
#                             'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                             'license_plate': {
#                                 'bbox': [x1, y1, x2, y2],
#                                 'text': best_text,
#                                 'bbox_score': score,
#                                 'text_score': best_score
#                             }
#                         }
#                     }
#                     print(f"Car {car_id}: Valid license plate '{best_text}' with score {best_score}")
#                 else:
#                     print(f"Car {car_id}: Invalid format '{best_text}'")
#             else:
#                 print(f"Car {car_id}: License text '{best_text}' has invalid length.")
#
# # Write CSV output
# write_csv(results, './test_image.csv')
# print("Processing complete. Results saved to test_image.csv")

# import cv2
# import pytesseract
# import csv
# import torch
# from pathlib import Path
# from datetime import datetime
#
# # Paths
# vehicle_model_path = 'yolo11n.pt'
# license_plate_model_path = '/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt'
# image_path = '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/frame-190.jpg'
# csv_output_path = 'license_plate_results.csv'
#
# # Load YOLO models
# vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom', path=vehicle_model_path)
# lp_model = torch.hub.load('ultralytics/yolov5', 'custom', path=license_plate_model_path)
#
# # Read image
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Detect vehicles
# vehicle_results = vehicle_model(image_rgb)
# vehicle_detections = vehicle_results.xyxy[0]  # format: [x1, y1, x2, y2, conf, cls]
#
# # Prepare CSV
# csv_header = ["frame_nmr", "car_id", "car_bbox", "license_plate_bbox", "license_plate_bbox_score", "license_number", "license_number_score"]
# csv_rows = []
#
# frame_id = 1  # Can be dynamic if using video frames
#
# # Process each vehicle
# car_id = 0
# for *car_box, car_conf, car_cls in vehicle_detections:
#     car_id += 1
#     x1, y1, x2, y2 = map(int, car_box)
#     car_crop = image_rgb[y1:y2, x1:x2]
#
#     # Detect license plates in car crop
#     lp_results = lp_model(car_crop)
#     lp_detections = lp_results.xyxy[0]
#
#     for *lp_box, lp_conf, lp_cls in lp_detections:
#         lx1, ly1, lx2, ly2 = map(int, lp_box)
#         license_crop = car_crop[ly1:ly2, lx1:lx2]
#
#         # OCR using Tesseract
#         license_gray = cv2.cvtColor(license_crop, cv2.COLOR_RGB2GRAY)
#         license_text = pytesseract.image_to_string(license_gray, config='--psm 7').strip()
#         confidence = 0.85 if license_text else 0.0
#
#         # Append to CSV row
#         csv_rows.append([
#             frame_id,
#             car_id,
#             [x1, y1, x2, y2],
#             [lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1],  # license plate bbox in original image coords
#             round(float(lp_conf), 2),
#             license_text,
#             confidence
#         ])
#
#         # Draw boxes
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # car box
#         cv2.rectangle(image, (lx1 + x1, ly1 + y1), (lx2 + x1, ly2 + y1), (0, 0, 255), 2)  # license plate box
#         cv2.putText(image, license_text, (lx1 + x1, ly1 + y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
# # Save annotated image
# cv2.imwrite("annotated_output.png", image)
#
# # Write to CSV
# with open(csv_output_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(csv_header)
#     writer.writerows(csv_rows)
#
# print(f"✅ Processing complete. Results saved to {csv_output_path}")


# from ultralytics import YOLO
# import cv2
# import numpy as np
# # from sort.sort import Sort # Not needed for single image
# from util import get_car, read_license_plate, write_csv  # Your util.py functions
#
# # --- Configuration ---
# IMAGE_PATH = '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/frame-190.jpg'  # <<< IMPORTANT: Set your image path here
# OUTPUT_CSV_PATH = './single_image_results.csv'
#
# # Verify your model paths
# COCO_MODEL_PATH = 'yolo11n.pt'  # Or 'yolov5n.pt', or your specific 'yolo11n.pt'
# LICENSE_PLATE_DETECTOR_PATH = '/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt'  # User's path
#
# VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
# LICENSE_PLATE_CONFIDENCE_THRESHOLD = 0.3  # Min confidence for detected license plates
#
# # --- Load models ---
# try:
#     coco_model = YOLO(COCO_MODEL_PATH)
#     license_plate_detector = YOLO(LICENSE_PLATE_DETECTOR_PATH)
# except Exception as e:
#     print(f"Error loading YOLO models: {e}")
#     print(
#         f"Please check model paths: COCO_MODEL_PATH='{COCO_MODEL_PATH}', LICENSE_PLATE_DETECTOR_PATH='{LICENSE_PLATE_DETECTOR_PATH}'")
#     exit()
#
# # --- Load the single image ---
# frame = cv2.imread(IMAGE_PATH)
# if frame is None:
#     print(f"Error: Could not read image from '{IMAGE_PATH}'")
#     exit()
#
# # --- Initialize results for the single image ---
# single_image_key = 'image_0'
# results = {single_image_key: {}}
#
# # --- 1. Detect Vehicles in the single frame ---
# vehicle_detections_yolo = coco_model(frame)[0]
# cars_for_get_car = []
# car_index_in_frame = 0
# for detection in vehicle_detections_yolo.boxes.data.tolist():
#     x1, y1, x2, y2, score, class_id = detection
#     if int(class_id) in VEHICLE_CLASS_IDS:
#         cars_for_get_car.append([x1, y1, x2, y2, car_index_in_frame])
#         car_index_in_frame += 1
#
# print(f"Detected {len(cars_for_get_car)} vehicles in the image.")
#
# # --- 2. Detect License Plates ---
# license_plate_detections_yolo = license_plate_detector(frame)[0]
# print(f"Detected {len(license_plate_detections_yolo.boxes.data.tolist())} potential license plates.")
#
# for lp_detection_data in license_plate_detections_yolo.boxes.data.tolist():
#     lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = lp_detection_data
#
#     if lp_score < LICENSE_PLATE_CONFIDENCE_THRESHOLD:
#         continue
#
#     if not cars_for_get_car:
#         print(
#             f"No vehicles detected to associate with license plate at [{lp_x1:.0f}, {lp_y1:.0f}, {lp_x2:.0f}, {lp_y2:.0f}].")
#         continue
#
#     # --- 3. Assign License Plate to a Car using util.get_car ---
#     # util.get_car expects license_plate as (x1, y1, x2, y2, score, class_id)
#     # and vehicle_track_ids as list of [xcar1, ycar1, xcar2, ycar2, car_id]
#     car_x1_ret, car_y1_ret, car_x2_ret, car_y2_ret, car_id_from_get_car = get_car(lp_detection_data, cars_for_get_car)
#
#     if car_id_from_get_car != -1:
#         # --- 4. Crop and Read License Plate Text ---
#         padding = 5
#         crop_lp_y1 = max(0, int(lp_y1) - padding)
#         crop_lp_y2 = min(frame.shape[0], int(lp_y2) + padding)
#         crop_lp_x1 = max(0, int(lp_x1) - padding)
#         crop_lp_x2 = min(frame.shape[1], int(lp_x2) + padding)
#
#         license_plate_crop = frame[crop_lp_y1:crop_lp_y2, crop_lp_x1:crop_lp_x2]
#
#         if license_plate_crop.size == 0:
#             print(
#                 f"Warning: Empty crop for license plate at [{lp_x1:.0f},{lp_y1:.0f},{lp_x2:.0f},{lp_y2:.0f}] for car {car_id_from_get_car}. Skipping.")
#             continue
#
#         license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#
#         # threshold_values = [64, 128, 180]
#         threshold_values = [10, 40, 80]
#         best_text_from_ocr = None
#         best_ocr_score = 0
#
#         for thresh_val in threshold_values:
#             # Apply threshold to the grayscale crop
#             _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, thresh_val, 255,
#                                                          cv2.THRESH_BINARY_INV)
#
#             # Call your util.read_license_plate with the thresholded (binary) image
#             # It returns (formatted_text, confidence_score) if successful
#             formatted_text, ocr_confidence = read_license_plate(license_plate_crop_thresh)
#
#             if formatted_text is not None and ocr_confidence is not None and ocr_confidence > best_ocr_score:
#                 best_text_from_ocr = formatted_text
#                 best_ocr_score = ocr_confidence
#
#         # If best_text_from_ocr is not None, it means util.read_license_plate returned a valid,
#         # formatted plate that passed its internal checks (format_license, license_complies_format).
#         if best_text_from_ocr is not None:
#             results[single_image_key][car_id_from_get_car] = {
#                 'car': {'bbox': [car_x1_ret, car_y1_ret, car_x2_ret, car_y2_ret]},
#                 'license_plate': {
#                     'bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
#                     'text': best_text_from_ocr,  # This is the validated, formatted text from util.py
#                     'bbox_score': lp_score,  # Detection score of the LP bounding box
#                     'text_score': best_ocr_score  # Confidence score from read_license_plate
#                 }
#             }
#             print(f"Car ID {car_id_from_get_car}: Valid LP '{best_text_from_ocr}', OCR Score: {best_ocr_score:.2f}")
#         else:
#             print(
#                 f"Car ID {car_id_from_get_car}: Could not read valid text from license plate at [{lp_x1:.0f},{lp_y1:.0f},{lp_x2:.0f},{lp_y2:.0f}] after trying thresholds.")
#     # else:
#     #     print(f"License plate at [{lp_x1:.0f},{lp_y1:.0f},{lp_x2:.0f},{lp_y2:.0f}] not associated with any car by util.get_car.")
#
# # --- Write results to CSV using util.write_csv ---
# try:
#     if results[single_image_key]:
#         write_csv(results, OUTPUT_CSV_PATH)
#         print(f"Processing complete. Results written to {OUTPUT_CSV_PATH}")
#         print(f"Found {len(results[single_image_key])} license plate(s) associated with cars in the image.")
#     else:
#         print(f"Processing complete. No valid license plates associated with cars were found to write to CSV.")
# except Exception as e:
#     print(f"Error writing CSV: {e}")
#     print("Current results structure (if any):", results)
#
# # Optional: Display image with detections for debugging
# # (You can uncomment this section if you want to see the output visually)
# display_frame = frame.copy()
# for car_id_res, data in results.get(single_image_key, {}).items():
#     car_bbox = data['car']['bbox']
#     lp_info = data['license_plate']
#     lp_bbox = lp_info['bbox']
#     lp_text = lp_info['text']
#     # Draw car bounding box (green)
#     cv2.rectangle(display_frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (0, 255, 0), 2)
#     # Draw license plate bounding box (blue)
#     cv2.rectangle(display_frame, (int(lp_bbox[0]), int(lp_bbox[1])), (int(lp_bbox[2]), int(lp_bbox[3])), (255, 0, 0), 2)
#     cv2.putText(display_frame, lp_text, (int(lp_bbox[0]), int(lp_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
#
# cv2.imshow(f"Detections in {IMAGE_PATH}", display_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Enhanced License Plate Detection with Perspective Transformation and Improved Preprocessing
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


# For image processing using Perspective Correction
#
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# from sort.sort import *
# from image_util import get_car, read_license_plate, write_csv
# # Import your perspective transformation functions
# from main import detect_corners_and_apply_perspective_transform, enhance_license_plate_image
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


# Example usage
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


from ultralytics import YOLO
import cv2
import numpy as np
import os
from sort.sort import *
from image_util import get_car, read_license_plate, write_csv


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


def process_image_core(frame, coco_model, license_plate_detector):
    """
    Core image processing logic that can be used for both single and batch processing
    """
    vehicles = [2, 3, 5, 7]  # Class IDs for vehicles
    image_results = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # For images, assign simple IDs to vehicles (no tracking needed)
    track_ids = []
    for i, detection in enumerate(detections_):
        x1, y1, x2, y2, score = detection
        track_ids.append([x1, y1, x2, y2, i])  # Use index as ID

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
            padding = 10
            y1_crop = max(0, int(y1) - padding)
            y2_crop = min(frame.shape[0], int(y2) + padding)
            x1_crop = max(0, int(x1) - padding)
            x2_crop = min(frame.shape[1], int(x2) + padding)

            # Crop license plate
            license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            # Apply enhanced preprocessing with perspective transformation
            enhanced_license_plate, transform_applied = enhance_license_plate_image(license_plate_crop)

            # Try multiple threshold values for better OCR
            threshold_values = [0, 64, 128, 180]
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
                        image_results[car_id] = {
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
                        print(f"Car {car_id}: Valid license plate '{best_text}' "
                              f"(format: XX00XX0000) with score {best_score} {transform_msg}")

    return image_results


def process_single_image(image_path, output_csv_path):
    """
    Process a single image for license plate detection and recognition
    """
    # Load models
    coco_model = YOLO('yolo11n.pt')
    license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    print(f"Processing single image: {image_path}")

    # Process the image
    image_results = process_image_core(frame, coco_model, license_plate_detector)

    # Format results for CSV (single image = frame 0)
    results = {0: image_results}

    # Write results
    write_csv(results, output_csv_path)
    print(f"Processing complete. Results written to {output_csv_path}")

    return results


def process_multiple_images(image_paths, output_csv_path):
    """
    Process multiple images and combine results
    """
    # Load models once for all images
    coco_model = YOLO('yolo11n.pt')
    license_plate_detector = YOLO('/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt')

    combined_results = {}

    for frame_idx, image_path in enumerate(image_paths):
        print(f"Processing image {frame_idx + 1}/{len(image_paths)}: {image_path}")

        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            continue

        # Process the image
        image_results = process_image_core(frame, coco_model, license_plate_detector)

        # Add to combined results with frame index
        if image_results:  # Only add if we found license plates
            combined_results[frame_idx] = image_results

    # Write combined results
    write_csv(combined_results, output_csv_path)
    print(f"Batch processing complete. Processed {len(image_paths)} images.")
    print(f"Found license plates in {len(combined_results)} images.")
    print(f"Results written to {output_csv_path}")

    return combined_results


def process_image_folder(folder_path, output_csv_path, image_extensions=None):
    """
    Process all images in a folder
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Get all image files from folder
    image_paths = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, filename))

    if not image_paths:
        print(f"No image files found in {folder_path}")
        return None

    print(f"Found {len(image_paths)} images in {folder_path}")

    # Process all images
    return process_multiple_images(image_paths, output_csv_path)


# Example usage functions (uncomment to use)
if __name__ == "__main__":
    # Example 1: Process a single image
    process_single_image('frame-190.jpg', 'single_image_results.csv')

    # Example 2: Process multiple specific images
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # process_multiple_images(image_list, 'multiple_images_results.csv')

    # Example 3: Process all images in a folder
    # process_image_folder('path/to/image/folder', 'folder_results.csv')

    print("Image processing module loaded. Use the functions to process images.")