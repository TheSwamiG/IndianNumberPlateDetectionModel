# import cv2
# import numpy as np
# import os
# import glob
# import json
# from collections import Counter
# import argparse
#
#
# class LicensePlateAnalyzer:
#     """Simple analyzer for license plate detection results"""
#
#     def __init__(self, input_dir, output_dir=None):
#         self.input_dir = input_dir
#         self.output_dir = output_dir or os.path.join(input_dir, 'analysis')
#         os.makedirs(self.output_dir, exist_ok=True)
#
#     def analyze_results(self):
#         """Analyze detection results and create summary"""
#         print("Analyzing detection results...")
#
#         # Load detection summary
#         summary_path = os.path.join(self.input_dir, 'detection_summary.txt')
#
#         if not os.path.exists(summary_path):
#             print("No detection summary found!")
#             return None
#
#         detections = self.parse_summary_file(summary_path)
#
#         if not detections:
#             print("No valid detections found!")
#             return None
#
#         # Analyze patterns
#         analysis = self.create_analysis(detections)
#
#         # Save analysis
#         self.save_analysis(analysis)
#
#         # Create enhanced crops view
#         self.create_crops_summary()
#
#         return analysis
#
#     def parse_summary_file(self, summary_path):
#         """Parse the detection summary file"""
#         detections = []
#
#         with open(summary_path, 'r') as f:
#             lines = f.readlines()
#
#         current_image = None
#
#         for line in lines:
#             line = line.strip()
#
#             if line.startswith('Image:'):
#                 current_image = line.split('Image:')[1].strip()
#             elif '. License Plate:' in line and current_image:
#                 # Extract license plate text
#                 parts = line.split('License Plate:')
#                 if len(parts) > 1:
#                     plate_text = parts[1].strip()
#                     detections.append({
#                         'image': current_image,
#                         'plate': plate_text,
#                         'state': plate_text[:2] if len(plate_text) >= 2 else '',
#                         'district': plate_text[2:4] if len(plate_text) >= 4 else ''
#                     })
#
#         return detections
#
#     def create_analysis(self, detections):
#         """Create analysis from detections"""
#         analysis = {
#             'total_detections': len(detections),
#             'unique_images': len(set(d['image'] for d in detections)),
#             'state_counts': dict(Counter(d['state'] for d in detections if d['state'])),
#             'district_counts': dict(Counter(d['district'] for d in detections if d['district'])),
#             'plate_lengths': dict(Counter(len(d['plate'].replace(' ', '')) for d in detections)),
#             'all_plates': [d['plate'] for d in detections]
#         }
#
#         return analysis
#
#     def save_analysis(self, analysis):
#         """Save analysis to JSON file"""
#         analysis_path = os.path.join(self.output_dir, 'analysis.json')
#
#         with open(analysis_path, 'w') as f:
#             json.dump(analysis, f, indent=2)
#
#         print(f"Analysis saved to: {analysis_path}")
#
#         # Also create a readable text report
#         report_path = os.path.join(self.output_dir, 'analysis_report.txt')
#
#         with open(report_path, 'w') as f:
#             f.write("LICENSE PLATE DETECTION ANALYSIS\n")
#             f.write("=" * 40 + "\n\n")
#
#             f.write(f"Total Detections: {analysis['total_detections']}\n")
#             f.write(f"Images Processed: {analysis['unique_images']}\n\n")
#
#             f.write("TOP STATES:\n")
#             top_states = sorted(analysis['state_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
#             for state, count in top_states:
#                 f.write(f"  {state}: {count}\n")
#
#             f.write("\nTOP DISTRICTS:\n")
#             top_districts = sorted(analysis['district_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
#             for district, count in top_districts:
#                 f.write(f"  {district}: {count}\n")
#
#             f.write("\nPLATE LENGTHS:\n")
#             for length, count in sorted(analysis['plate_lengths'].items()):
#                 f.write(f"  {length} characters: {count}\n")
#
#             f.write("\nALL DETECTED PLATES:\n")
#             for i, plate in enumerate(analysis['all_plates'], 1):
#                 f.write(f"  {i:3d}. {plate}\n")
#
#         print(f"Report saved to: {report_path}")
#
#     def create_crops_summary(self):
#         """Create a summary view of all detected license plate crops"""
#         crops_dir = os.path.join(self.input_dir, 'crops')
#
#         if not os.path.exists(crops_dir):
#             print("No crops directory found!")
#             return
#
#         crop_files = glob.glob(os.path.join(crops_dir, '*.jpg'))
#
#         if not crop_files:
#             print("No crop images found!")
#             return
#
#         print(f"Found {len(crop_files)} license plate crops")
#
#         # Create a grid view of all crops
#         self.create_crops_grid(crop_files)
#
#     def create_crops_grid(self, crop_files):
#         """Create a grid view of license plate crops"""
#         if not crop_files:
#             return
#
#         # Load all crops
#         crops = []
#         labels = []
#
#         for crop_file in sorted(crop_files):
#             img = cv2.imread(crop_file)
#             if img is not None:
#                 # Resize to standard size
#                 img_resized = cv2.resize(img, (200, 100))
#                 crops.append(img_resized)
#
#                 # Extract label from filename or image
#                 filename = os.path.basename(crop_file)
#                 labels.append(filename)
#
#         if not crops:
#             return
#
#         # Calculate grid dimensions
#         cols = min(4, len(crops))  # Maximum 4 columns
#         rows = (len(crops) + cols - 1) // cols
#
#         # Create grid image
#         grid_width = cols * 200
#         grid_height = rows * 120  # Extra space for labels
#
#         grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
#
#         for i, (crop, label) in enumerate(zip(crops, labels)):
#             row = i // cols
#             col = i % cols
#
#             y_start = row * 120
#             x_start = col * 200
#
#             # Place crop
#             grid_img[y_start:y_start + 100, x_start:x_start + 200] = crop
#
#             # Add label
#             cv2.putText(grid_img, label[:20], (x_start + 5, y_start + 115),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
#
#         # Save grid
#         grid_path = os.path.join(self.output_dir, 'crops_grid.jpg')
#         cv2.imwrite(grid_path, grid_img)
#
#         print(f"Crops grid saved to: {grid_path}")
#
#     def enhance_detection_images(self):
#         """Apply simple enhancements to annotated images"""
#         print("Enhancing annotated images...")
#
#         # Find annotated images
#         annotated_images = glob.glob(os.path.join(self.input_dir, '*_annotated.jpg'))
#
#         if not annotated_images:
#             print("No annotated images found!")
#             return
#
#         enhanced_dir = os.path.join(self.output_dir, 'enhanced')
#         os.makedirs(enhanced_dir, exist_ok=True)
#
#         for img_path in annotated_images:
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue
#
#             # Simple enhancement: adjust brightness and contrast
#             enhanced = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
#
#             # Reduce noise
#             enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
#
#             # Save enhanced image
#             filename = os.path.basename(img_path)
#             enhanced_path = os.path.join(enhanced_dir, f"enhanced_{filename}")
#             cv2.imwrite(enhanced_path, enhanced)
#
#         print(f"Enhanced images saved to: {enhanced_dir}")
#
#     def process_all(self):
#         """Run complete analysis"""
#         print(f"Starting analysis for: {self.input_dir}")
#         print(f"Output directory: {self.output_dir}")
#
#         # Analyze results
#         analysis = self.analyze_results()
#
#         # Enhance images
#         self.enhance_detection_images()
#
#         print(f"\nAnalysis complete!")
#         print(f"Check output directory: {self.output_dir}")
#
#         if analysis:
#             print(f"\nQuick Summary:")
#             print(f"- Total detections: {analysis['total_detections']}")
#             print(f"- Unique states: {len(analysis['state_counts'])}")
#             print(
#                 f"- Most common state: {max(analysis['state_counts'].items(), key=lambda x: x[1])[0] if analysis['state_counts'] else 'None'}")
#
#         return analysis
#
#
# def main():
#     parser = argparse.ArgumentParser(description='Analyze license plate detection results')
#     parser.add_argument('--input', '-i', required=True,
#                         help='Input directory containing detection results')
#     parser.add_argument('--output', '-o', default=None,
#                         help='Output directory for analysis (default: input_dir/analysis)')
#
#     args = parser.parse_args()
#
#     if not os.path.exists(args.input):
#         print(f"Error: Input directory {args.input} does not exist")
#         return
#
#     # Run analysis
#     analyzer = LicensePlateAnalyzer(args.input, args.output)
#     analyzer.process_all()
#
#
# if __name__ == "__main__":
#     main()

# import ast
# import cv2
# import numpy as np
# import pandas as pd
# import os
# import glob
#
#
# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
#     x1, y1 = top_left
#     x2, y2 = bottom_right
#
#     cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
#     cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
#
#     cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
#     cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
#
#     cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
#     cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
#
#     cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
#     cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
#
#     return img
#
#
# def safe_ast_eval(bbox_str):
#     """Safely parse bounding box string with multiple format handling"""
#     try:
#         # Clean the string
#         cleaned = bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
#         return ast.literal_eval(cleaned)
#     except:
#         try:
#             # Alternative parsing method
#             cleaned = bbox_str.replace('[', '').replace(']', '').replace(',', ' ')
#             coords = [float(x) for x in cleaned.split()]
#             return coords
#         except:
#             print(f"Warning: Could not parse bbox string: {bbox_str}")
#             return [0, 0, 100, 100]  # Default fallback
#
#
# # Configuration - UPDATE THESE PATHS
# csv_file = './single_image_results.csv'  # Path to your CSV file with annotations
# images_folder = './images/'  # Folder containing input images
# output_folder = './annotated_images/'  # Folder to save annotated images
#
# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)
#
# # Load results
# try:
#     results = pd.read_csv(csv_file)
#     print(f"Loaded {len(results)} annotations from {csv_file}")
# except FileNotFoundError:
#     print(f"Error: Could not find CSV file at {csv_file}")
#     exit()
#
# # Get list of unique frame numbers/image names from CSV
# unique_frames = results['frame_nmr'].unique()
# print(f"Found annotations for {len(unique_frames)} images")
#
# # Extract best license plate for each car across all images
# license_plate = {}
# for car_id in results['car_id'].unique():
#     car_data = results[results['car_id'] == car_id]
#
#     # Find the frame with the highest license number score for this car
#     max_score_idx = car_data['license_number_score'].astype(float).idxmax()
#     best_row = results.loc[max_score_idx]
#
#     license_plate[car_id] = {
#         'license_crop': None,
#         'license_plate_number': best_row['license_number'],
#         'best_frame': best_row['frame_nmr']
#     }
#
# print(f"Processing {len(license_plate)} unique cars")
#
# # List all image files in the folder to help with debugging
# print(f"\nLooking for images in: {images_folder}")
# if os.path.exists(images_folder):
#     image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
#     print(f"Found {len(image_files)} image files: {image_files}")
# else:
#     print(f"Images folder does not exist: {images_folder}")
#
# # Process each image
# processed_count = 0
# for frame_nmr in unique_frames:
#     print(f"\nLooking for image for frame {frame_nmr}")
#
#     # Try multiple naming patterns
#     patterns = [
#         f"frame-{frame_nmr}.jpg",
#         f"{frame_nmr}.jpg",
#         f"image_{frame_nmr}.jpg",
#         f"img_{frame_nmr}.jpg",
#         f"frame_{frame_nmr}.png",
#         f"frame-{frame_nmr}.png",
#         f"image_{frame_nmr}.png",
#         f"img_{frame_nmr}.png",
#         f"frame{frame_nmr}.jpg",
#         f"frame{frame_nmr}.png"
#     ]
#
#     # If frame_nmr is 0, also try some common single image names
#     if frame_nmr == 0:
#         patterns.extend([
#             "test.jpg", "test.png", "sample.jpg", "sample.png",
#             "input.jpg", "input.png", "image.jpg", "image.png"
#         ])
#
#     image_path = None
#     for pattern in patterns:
#         test_path = os.path.join(images_folder, pattern)
#         print(f"  Trying: {pattern}")
#         if os.path.exists(test_path):
#             image_path = test_path
#             print(f"  Found: {pattern}")
#             break
#
#     # If still not found, try to match any image file if there's only one
#     if image_path is None and os.path.exists(images_folder):
#         image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
#         if len(image_files) == 1:
#             image_path = os.path.join(images_folder, image_files[0])
#             print(f"  Using single image file: {image_files[0]}")
#         elif len(image_files) > 1:
#             print(f"  Multiple image files found, please specify which one to use:")
#             for i, img_file in enumerate(image_files):
#                 print(f"    {i}: {img_file}")
#
#     if image_path is None:
#         print(f"Warning: Could not find image file for frame {frame_nmr}")
#         print(f"Available files in {images_folder}:")
#         if os.path.exists(images_folder):
#             all_files = os.listdir(images_folder)
#             for file in all_files:
#                 print(f"  {file}")
#         continue
#
#     # Load image
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"Warning: Could not load image {image_path}")
#         continue
#
#     height, width = frame.shape[:2]
#     print(f"Processing frame {frame_nmr}: {width}x{height}")
#
#     # Get detections for current frame
#     df_ = results[results['frame_nmr'] == frame_nmr]
#
#     # Extract license plate crops for cars that have their best frame in this image
#     for car_id in license_plate.keys():
#         if license_plate[car_id]['best_frame'] == frame_nmr and license_plate[car_id]['license_crop'] is None:
#             # Find this car's data in current frame
#             car_data = df_[df_['car_id'] == car_id]
#             if len(car_data) > 0:
#                 best_row = car_data.iloc[0]
#
#                 # Parse bounding box coordinates using safe method
#                 coords = safe_ast_eval(best_row['license_plate_bbox'])
#                 x1, y1, x2, y2 = [int(c) for c in coords]
#
#                 # Ensure coordinates are within frame bounds
#                 x1 = max(0, min(x1, width))
#                 y1 = max(0, min(y1, height))
#                 x2 = max(0, min(x2, width))
#                 y2 = max(0, min(y2, height))
#
#                 if x2 > x1 and y2 > y1:  # Valid bounding box
#                     license_crop = frame[y1:y2, x1:x2, :]
#
#                     if license_crop.size > 0:
#                         # Resize license plate crop for better visibility - MUCH LARGER
#                         crop_height = 160  # Doubled height for better visibility
#                         crop_width = int((x2 - x1) * crop_height / (y2 - y1)) if (y2 - y1) > 0 else 400
#                         crop_width = max(crop_width, 300)  # Increased minimum width
#
#                         try:
#                             license_crop = cv2.resize(license_crop, (crop_width, crop_height))
#                             license_plate[car_id]['license_crop'] = license_crop
#                         except:
#                             print(f"Warning: Could not resize license crop for car {car_id}")
#
#     # Process annotations for current frame
#     for row_idx in range(len(df_)):
#         row = df_.iloc[row_idx]
#         car_id = row['car_id']
#
#         # Parse car bounding box using safe method
#         car_coords = safe_ast_eval(row['car_bbox'])
#         car_x1, car_y1, car_x2, car_y2 = [int(c) for c in car_coords]
#
#         # Ensure car coordinates are within bounds
#         car_x1 = max(0, min(car_x1, width))
#         car_y1 = max(0, min(car_y1, height))
#         car_x2 = max(0, min(car_x2, width))
#         car_y2 = max(0, min(car_y2, height))
#
#         # Draw car bounding box
#         if car_x2 > car_x1 and car_y2 > car_y1:
#             draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25,
#                         line_length_x=200, line_length_y=200)
#
#         # Parse license plate bounding box using safe method
#         lp_coords = safe_ast_eval(row['license_plate_bbox'])
#         lp_x1, lp_y1, lp_x2, lp_y2 = [int(c) for c in lp_coords]
#
#         # Ensure license plate coordinates are within bounds
#         lp_x1 = max(0, min(lp_x1, width))
#         lp_y1 = max(0, min(lp_y1, height))
#         lp_x2 = max(0, min(lp_x2, width))
#         lp_y2 = max(0, min(lp_y2, height))
#
#         # Draw license plate bounding box
#         if lp_x2 > lp_x1 and lp_y2 > lp_y1:
#             cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 12)
#
#         # Add license plate crop and text if available
#         if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
#             license_crop = license_plate[car_id]['license_crop']
#             license_text = license_plate[car_id]['license_plate_number']
#
#             H, W = license_crop.shape[:2]
#
#             # Calculate position for license plate display (above the car) - MORE SPACE
#             display_y = max(180, car_y1 - 220)  # More space from top and car
#             center_x = (car_x1 + car_x2) // 2
#             display_x1 = max(0, center_x - W // 2)
#             display_x2 = min(width, display_x1 + W)
#
#             # Adjust if the crop would go outside frame bounds
#             if display_x2 >= width:
#                 display_x1 = width - W
#                 display_x2 = width
#             if display_x1 < 0:
#                 display_x1 = 0
#                 display_x2 = W
#
#             # Place license plate crop
#             try:
#                 if (display_y + H < height and
#                         display_x1 >= 0 and display_x2 <= width and
#                         display_y >= 0):
#
#                     frame[display_y:display_y + H, display_x1:display_x2, :] = license_crop
#
#                     # Add white background for text - LARGER
#                     text_bg_y1 = max(0, display_y - 80)  # Larger text background
#                     text_bg_y2 = display_y
#                     if text_bg_y2 > text_bg_y1:
#                         frame[text_bg_y1:text_bg_y2, display_x1:display_x2, :] = (255, 255, 255)
#
#                     # Add license plate text - MUCH LARGER
#                     if license_text and license_text != '0':
#                         font_scale = 2.5  # Much larger font
#                         thickness = 6  # Thicker text
#                         (text_width, text_height), _ = cv2.getTextSize(
#                             license_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#
#                         text_x = max(0, center_x - text_width // 2)
#                         text_y = max(text_height, display_y - 20)  # More space from license plate
#
#                         cv2.putText(frame, license_text,
#                                     (text_x, text_y),
#                                     cv2.FONT_HERSHEY_SIMPLEX,
#                                     font_scale, (0, 0, 0), thickness)
#
#             except Exception as e:
#                 print(f"Warning: Could not place license crop for car {car_id} at frame {frame_nmr}: {e}")
#                 continue
#
#     # Save annotated image
#     output_filename = f"annotated_frame-{frame_nmr}.jpg"
#     output_path = os.path.join(output_folder, output_filename)
#
#     if cv2.imwrite(output_path, frame):
#         processed_count += 1
#         print(f"Saved annotated image: {output_path}")
#     else:
#         print(f"Error: Could not save image {output_path}")
#
# print(f"\nImage processing complete!")
# print(f"Total images processed: {processed_count}")
# print(f"Annotated images saved in: {output_folder}")
#
# # Optional: Create a summary image showing all license plates found
# if license_plate:
#     print("\nCreating license plate summary...")
#
#     # Calculate grid dimensions for showing all license plates
#     num_plates = len([lp for lp in license_plate.values() if lp['license_crop'] is not None])
#     if num_plates > 0:
#         cols = min(4, num_plates)  # Max 4 columns
#         rows = (num_plates + cols - 1) // cols
#
#         # Create summary image
#         plate_height = 160
#         plate_width = 400
#         margin = 20
#
#         summary_width = cols * plate_width + (cols + 1) * margin
#         summary_height = rows * (plate_height + 60) + (rows + 1) * margin  # Extra space for text
#
#         summary_img = np.ones((summary_height, summary_width, 3), dtype=np.uint8) * 255
#
#         plate_idx = 0
#         for car_id, plate_info in license_plate.items():
#             if plate_info['license_crop'] is not None:
#                 row = plate_idx // cols
#                 col = plate_idx % cols
#
#                 y_start = margin + row * (plate_height + 60 + margin)
#                 x_start = margin + col * (plate_width + margin)
#
#                 # Resize license plate to standard size
#                 resized_plate = cv2.resize(plate_info['license_crop'], (plate_width, plate_height))
#
#                 # Place license plate
#                 summary_img[y_start:y_start + plate_height, x_start:x_start + plate_width] = resized_plate
#
#                 # Add car ID and license number text
#                 car_text = f"Car ID: {car_id}"
#                 license_text = f"License: {plate_info['license_plate_number']}"
#
#                 cv2.putText(summary_img, car_text, (x_start, y_start + plate_height + 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
#                 cv2.putText(summary_img, license_text, (x_start, y_start + plate_height + 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#                 plate_idx += 1
#
#         # Save summary image
#         summary_path = os.path.join(output_folder, "license_plates_summary.jpg")
#         if cv2.imwrite(summary_path, summary_img):
#             print(f"License plate summary saved: {summary_path}")
#         else:
#             print("Error: Could not save license plate summary")

import ast
import cv2
import numpy as np
import pandas as pd
import os
import glob


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def safe_ast_eval(bbox_str):
    """Safely parse bounding box string with multiple format handling"""
    try:
        # Clean the string
        cleaned = bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
        return ast.literal_eval(cleaned)
    except:
        try:
            # Alternative parsing method
            cleaned = bbox_str.replace('[', '').replace(']', '').replace(',', ' ')
            coords = [float(x) for x in cleaned.split()]
            return coords
        except:
            print(f"Warning: Could not parse bbox string: {bbox_str}")
            return [0, 0, 100, 100]  # Default fallback


def find_image_file(image_path_or_directory, frame_nmr=0):
    """
    Find the image file given a path or directory
    Returns the actual image path if found, None otherwise
    """
    # If it's a direct file path and exists
    if os.path.isfile(image_path_or_directory):
        return image_path_or_directory

    # If it's a directory, search for image files
    if os.path.isdir(image_path_or_directory):
        # Try multiple naming patterns
        patterns = [
            f"frame-{frame_nmr}.jpg",
            f"{frame_nmr}.jpg",
            f"image_{frame_nmr}.jpg",
            f"img_{frame_nmr}.jpg",
            f"frame_{frame_nmr}.png",
            f"frame-{frame_nmr}.png",
            f"image_{frame_nmr}.png",
            f"img_{frame_nmr}.png",
            f"frame{frame_nmr}.jpg",
            f"frame{frame_nmr}.png"
        ]

        # If frame_nmr is 0, also try common single image names
        if frame_nmr == 0:
            patterns.extend([
                "test.jpg", "test.png", "sample.jpg", "sample.png",
                "input.jpg", "input.png", "image.jpg", "image.png"
            ])

        for pattern in patterns:
            test_path = os.path.join(image_path_or_directory, pattern)
            if os.path.exists(test_path):
                return test_path

        # If still not found, try to match any image file if there's only one
        image_files = [f for f in os.listdir(image_path_or_directory)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(image_files) == 1:
            return os.path.join(image_path_or_directory, image_files[0])
        elif len(image_files) > 1:
            print(f"Multiple image files found in {image_path_or_directory}:")
            for i, img_file in enumerate(image_files):
                print(f"  {i}: {img_file}")
            print("Please specify which image to use or provide direct path")

    return None


def process_single_image_visualization(csv_file_path, image_path, output_path=None):
    """
    Process visualization for a single image using CSV results

    Args:
        csv_file_path: Path to the CSV file with detection results
        image_path: Path to the input image file or directory containing image
        output_path: Path for the output annotated image (optional)
    """

    # Load results from CSV
    try:
        results = pd.read_csv(csv_file_path)
        print(f"Loaded {len(results)} annotations from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    if len(results) == 0:
        print("No detection results found in CSV file")
        return None

    # Get the frame number (usually 0 for single image)
    frame_nmr = results['frame_nmr'].iloc[0]

    # Find the actual image file
    actual_image_path = find_image_file(image_path, frame_nmr)

    if actual_image_path is None:
        print(f"Error: Could not find image file")
        print(f"Searched in: {image_path}")
        if os.path.isdir(image_path):
            print("Available files:")
            for file in os.listdir(image_path):
                print(f"  {file}")
        return None

    # Load the image
    frame = cv2.imread(actual_image_path)
    if frame is None:
        print(f"Error: Could not load image from {actual_image_path}")
        return None

    print(f"Successfully loaded image: {actual_image_path}")
    height, width = frame.shape[:2]
    print(f"Image dimensions: {width}x{height}")

    # Extract best license plate for each car
    license_plate_data = {}
    for car_id in results['car_id'].unique():
        car_data = results[results['car_id'] == car_id]

        # Find the detection with the highest license number score for this car
        max_score_idx = car_data['license_number_score'].astype(float).idxmax()
        best_row = results.loc[max_score_idx]

        license_plate_data[car_id] = {
            'license_crop': None,
            'license_plate_number': best_row['license_number'],
            'car_bbox': best_row['car_bbox'],
            'license_plate_bbox': best_row['license_plate_bbox'],
            'score': best_row['license_number_score']
        }

    print(f"Processing {len(license_plate_data)} unique cars")

    # Extract license plate crops
    for car_id, plate_info in license_plate_data.items():
        # Parse license plate bounding box
        lp_coords = safe_ast_eval(plate_info['license_plate_bbox'])
        x1, y1, x2, y2 = [int(c) for c in lp_coords]

        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 > x1 and y2 > y1:  # Valid bounding box
            license_crop = frame[y1:y2, x1:x2]

            if license_crop.size > 0:
                # Resize license plate crop for better visibility
                crop_height = 160
                crop_width = int((x2 - x1) * crop_height / (y2 - y1)) if (y2 - y1) > 0 else 400
                crop_width = max(crop_width, 300)

                try:
                    license_crop_resized = cv2.resize(license_crop, (crop_width, crop_height))
                    license_plate_data[car_id]['license_crop'] = license_crop_resized
                    print(f"Extracted license plate crop for car {car_id}")
                except Exception as e:
                    print(f"Warning: Could not resize license crop for car {car_id}: {e}")

    # Create annotated image
    annotated_frame = frame.copy()

    # Process each detection
    for car_id, plate_info in license_plate_data.items():
        # Parse car bounding box
        car_coords = safe_ast_eval(plate_info['car_bbox'])
        car_x1, car_y1, car_x2, car_y2 = [int(c) for c in car_coords]

        # Ensure car coordinates are within bounds
        car_x1 = max(0, min(car_x1, width))
        car_y1 = max(0, min(car_y1, height))
        car_x2 = max(0, min(car_x2, width))
        car_y2 = max(0, min(car_y2, height))

        # Draw car bounding box with decorative corners
        if car_x2 > car_x1 and car_y2 > car_y1:
            draw_border(annotated_frame, (car_x1, car_y1), (car_x2, car_y2),
                        (0, 255, 0), 25, line_length_x=200, line_length_y=200)

        # Parse and draw license plate bounding box
        lp_coords = safe_ast_eval(plate_info['license_plate_bbox'])
        lp_x1, lp_y1, lp_x2, lp_y2 = [int(c) for c in lp_coords]

        # Ensure license plate coordinates are within bounds
        lp_x1 = max(0, min(lp_x1, width))
        lp_y1 = max(0, min(lp_y1, height))
        lp_x2 = max(0, min(lp_x2, width))
        lp_y2 = max(0, min(lp_y2, height))

        # Draw license plate bounding box
        if lp_x2 > lp_x1 and lp_y2 > lp_y1:
            cv2.rectangle(annotated_frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 12)

        # Add license plate crop and text if available
        if plate_info['license_crop'] is not None:
            license_crop = plate_info['license_crop']
            license_text = plate_info['license_plate_number']

            H, W = license_crop.shape[:2]

            # Calculate position for license plate display (above the car)
            display_y = max(180, car_y1 - 220)
            center_x = (car_x1 + car_x2) // 2
            display_x1 = max(0, center_x - W // 2)
            display_x2 = min(width, display_x1 + W)

            # Adjust if the crop would go outside frame bounds
            if display_x2 >= width:
                display_x1 = width - W
                display_x2 = width
            if display_x1 < 0:
                display_x1 = 0
                display_x2 = W

            # Place license plate crop
            try:
                if (display_y + H < height and
                        display_x1 >= 0 and display_x2 <= width and
                        display_y >= 0):

                    annotated_frame[display_y:display_y + H, display_x1:display_x2] = license_crop

                    # Add white background for text
                    text_bg_y1 = max(0, display_y - 80)
                    text_bg_y2 = display_y
                    if text_bg_y2 > text_bg_y1:
                        cv2.rectangle(annotated_frame,
                                      (display_x1, text_bg_y1),
                                      (display_x2, text_bg_y2),
                                      (255, 255, 255), -1)

                    # Add license plate text
                    if license_text and license_text != '0':
                        font_scale = 2.5
                        thickness = 6
                        (text_width, text_height), _ = cv2.getTextSize(
                            license_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                        text_x = max(0, center_x - text_width // 2)
                        text_y = max(text_height, display_y - 20)

                        cv2.putText(annotated_frame, license_text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (0, 0, 0), thickness)

                        print(f"Added license plate text: {license_text} for car {car_id}")

                # Add car ID label
                car_label = f"Car {car_id}"
                cv2.putText(annotated_frame, car_label,
                            (car_x1, car_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 3)

            except Exception as e:
                print(f"Warning: Could not place license crop for car {car_id}: {e}")
                continue

    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(actual_image_path))[0]
        output_path = f"{base_name}_annotated.jpg"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save annotated image
    if cv2.imwrite(output_path, annotated_frame):
        print(f"✓ Annotated image saved: {output_path}")

        # Create summary
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Input image: {actual_image_path}")
        print(f"CSV results: {csv_file_path}")
        print(f"Output image: {output_path}")
        print(f"Cars detected: {len(license_plate_data)}")
        print(
            f"License plates detected: {len([p for p in license_plate_data.values() if p['license_crop'] is not None])}")

        # List all detected license plates
        for car_id, plate_info in license_plate_data.items():
            score = plate_info['score']
            text = plate_info['license_plate_number']
            print(f"  Car {car_id}: {text} (confidence: {score:.3f})")

        return output_path
    else:
        print(f"✗ Error: Could not save annotated image to {output_path}")
        return None


if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    csv_file = './single_image_results.csv'  # Path to your CSV file with detection results
    image_input = '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/frame-190.jpg'  # Direct path to image file
    output_image = './annotated_result.jpg'  # Output path for annotated image

    # Process the image
    result_path = process_single_image_visualization(
        csv_file_path=csv_file,
        image_path=image_input,
        output_path=output_image
    )

    if result_path:
        print(f"\n🎉 SUCCESS! Check your annotated image at: {result_path}")
    else:
        print(f"\n❌ FAILED! Please check the error messages above.")

    print("\n" + "=" * 50)
    print("TROUBLESHOOTING TIPS:")
    print("1. Make sure your main.py has generated the CSV file")
    print("2. Verify the image path points to the correct file")
    print("3. Check that the CSV contains detection results")
    print("4. Ensure you have write permissions for the output directory")