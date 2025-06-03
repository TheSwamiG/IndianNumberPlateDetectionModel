# Correct Version
# import ast
#
# import cv2
# import numpy as np
# import pandas as pd
#
#
# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
#     x1, y1 = top_left
#     x2, y2 = bottom_right
#
#     cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
#     cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
#
#     cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
#     cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
#
#     cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
#     cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
#
#     cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
#     cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
#
#     return img
#
#
# results = pd.read_csv('./test.csv')
#
# # load video
# video_path = 'IndianSample.mp4'
# cap = cv2.VideoCapture(video_path)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
#
# license_plate = {}
# for car_id in np.unique(results['car_id']):
#     max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
#     license_plate[car_id] = {'license_crop': None,
#                              'license_plate_number': results[(results['car_id'] == car_id) &
#                                                              (results['license_number_score'] == max_)]['license_number'].iloc[0]}
#     cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
#                                              (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
#     ret, frame = cap.read()
#
#     x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
#                                               (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
#
#     license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#     license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
#
#     license_plate[car_id]['license_crop'] = license_crop
#
#
# frame_nmr = -1
#
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# # read frames
# ret = True
# while ret:
#     ret, frame = cap.read()
#     frame_nmr += 1
#     if ret:
#         df_ = results[results['frame_nmr'] == frame_nmr]
#         for row_indx in range(len(df_)):
#             # draw car
#             car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
#             draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
#                         line_length_x=200, line_length_y=200)
#
#             # draw license plate
#             x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
#
#             # crop license plate
#             license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
#
#             H, W, _ = license_crop.shape
#
#             try:
#                 frame[int(car_y1) - H - 100:int(car_y1) - 100,
#                       int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
#
#                 frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
#                       int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)
#
#                 (text_width, text_height), _ = cv2.getTextSize(
#                     license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     4.3,
#                     17)
#
#                 cv2.putText(frame,
#                             license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
#                             (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             4.3,
#                             (0, 0, 0),
#                             17)
#
#             except:
#                 pass
#
#         out.write(frame)
#         frame = cv2.resize(frame, (1280, 720))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

# out.release()
# cap.release()
# import ast
#
# import cv2
# import numpy as np
# import pandas as pd
#
#
# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
#     x1, y1 = top_left
#     x2, y2 = bottom_right
#
#     cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
#     cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
#
#     cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
#     cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
#
#     cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
#     cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
#
#     cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
#     cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
#
#     return img
#
#
# def safe_bbox_parse(bbox_str):
#     """Safely parse bbox string, handling NaN and invalid values"""
#     if pd.isna(bbox_str) or not isinstance(bbox_str, str):
#         return None
#
#     try:
#         # Clean the bbox string
#         clean_bbox = str(bbox_str).replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
#         return ast.literal_eval(clean_bbox)
#     except:
#         return None
#
#
# results = pd.read_csv('./test_interpolated.csv')
#
# # load video
# video_path = 'Sample4.mp4'
# cap = cv2.VideoCapture(video_path)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
#
# license_plate = {}
#
# # Separate vehicle entries from standalone license plates
# vehicle_results = results[results['car_bbox'] != 'N/A']
# standalone_results = results[results['car_bbox'] == 'N/A']
#
# # Process vehicle entries (original logic)
# for car_id in vehicle_results['car_id'].unique():
#     try:
#         # Ensure car_id is numeric for vehicles
#         numeric_car_id = int(float(car_id))
#     except ValueError:
#         continue
#
#     car_data = vehicle_results[vehicle_results['car_id'] == car_id]
#     max_score = car_data['license_number_score'].astype(float).max()
#     best_detection = car_data[car_data['license_number_score'].astype(float) == max_score].iloc[0]
#
#     license_plate[car_id] = {
#         'license_crop': None,
#         'license_plate_number': best_detection['license_number']
#     }
#
#     # Get the frame for the best detection
#     cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_detection['frame_nmr']))
#     ret, frame = cap.read()
#
#     if ret:
#         bbox_coords = safe_bbox_parse(best_detection['license_plate_bbox'])
#         if bbox_coords:
#             try:
#                 x1, y1, x2, y2 = bbox_coords
#                 license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#                 if license_crop.size > 0:
#                     license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
#                     license_plate[car_id]['license_crop'] = license_crop
#             except:
#                 # If cropping fails, create a placeholder
#                 license_plate[car_id]['license_crop'] = np.ones((400, 200, 3), dtype=np.uint8) * 255
#         else:
#             # If bbox parsing fails, create a placeholder
#             license_plate[car_id]['license_crop'] = np.ones((400, 200, 3), dtype=np.uint8) * 255
#
# # Process standalone license plates
# standalone_license_plates = {}
# for idx, row in standalone_results.iterrows():
#     plate_id = row['car_id']
#     standalone_license_plates[plate_id] = {
#         'license_plate_number': row['license_number'],
#         'frame_nmr': int(row['frame_nmr']),
#         'bbox': row['license_plate_bbox']
#     }
#
# frame_nmr = -1
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# # read frames
# ret = True
# while ret:
#     ret, frame = cap.read()
#     frame_nmr += 1
#     if ret:
#         # Process vehicle detections
#         df_vehicles = vehicle_results[vehicle_results['frame_nmr'] == frame_nmr]
#         for row_indx in range(len(df_vehicles)):
#             row = df_vehicles.iloc[row_indx]
#
#             try:
#                 # draw car - safely parse car bbox
#                 car_bbox_coords = safe_bbox_parse(row['car_bbox'])
#                 if car_bbox_coords:
#                     car_x1, car_y1, car_x2, car_y2 = car_bbox_coords
#                     draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
#                                 line_length_x=200, line_length_y=200)
#
#                     # draw license plate - safely parse license plate bbox
#                     lp_bbox_coords = safe_bbox_parse(row['license_plate_bbox'])
#                     if lp_bbox_coords:
#                         x1, y1, x2, y2 = lp_bbox_coords
#                         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
#
#                         # Show license plate crop and text
#                         car_id = row['car_id']
#                         if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
#                             license_crop = license_plate[car_id]['license_crop']
#                             H, W, _ = license_crop.shape
#
#                             try:
#                                 # Place license crop above the car
#                                 frame[int(car_y1) - H - 100:int(car_y1) - 100,
#                                 int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
#
#                                 # White background for text
#                                 frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
#                                 int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)
#
#                                 # Add license plate text
#                                 license_text = str(license_plate[car_id]['license_plate_number'])
#                                 (text_width, text_height), _ = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX,
#                                                                                4.3, 17)
#
#                                 cv2.putText(frame, license_text,
#                                             (int((car_x2 + car_x1 - text_width) / 2),
#                                              int(car_y1 - H - 250 + (text_height / 2))),
#                                             cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
#                             except:
#                                 pass
#             except Exception as e:
#                 print(f"Error processing vehicle detection: {e}")
#                 continue
#
#         # Process standalone license plates
#         df_standalone = standalone_results[standalone_results['frame_nmr'] == frame_nmr]
#         for idx, row in df_standalone.iterrows():
#             try:
#                 # Draw standalone license plate - safely parse bbox
#                 lp_bbox_coords = safe_bbox_parse(row['license_plate_bbox'])
#                 if lp_bbox_coords:
#                     x1, y1, x2, y2 = lp_bbox_coords
#
#                     # Draw license plate with different color (purple) to distinguish from vehicle-associated plates
#                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 12)
#
#                     # Add license plate text near the detection
#                     license_text = str(row['license_number']) if pd.notna(row['license_number']) else ""
#                     if license_text and license_text != '0':
#                         cv2.putText(frame, license_text,
#                                     (int(x1), int(y1) - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
#             except Exception as e:
#                 print(f"Error processing standalone license plate: {e}")
#                 continue
#
#         out.write(frame)
#         frame = cv2.resize(frame, (1280, 720))
#
#         # Uncomment to show frame by frame
#         # cv2.imshow('frame', frame)
#         # cv2.waitKey(0)
#
# out.release()
# cap.release()
# import ast
# import cv2
# import numpy as np
# import pandas as pd
#
#
# def safe_bbox_parse(bbox_str):
#     """Safely parse bbox string, handling NaN and invalid values"""
#     if pd.isna(bbox_str) or not isinstance(bbox_str, str):
#         return None
#
#     try:
#         # Clean the bbox string
#         clean_bbox = str(bbox_str).replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
#         return ast.literal_eval(clean_bbox)
#     except:
#         return None
#
#
# def draw_text_with_background(img, text, position, font_scale=2.0, font_thickness=3,
#                               text_color=(0, 0, 0), bg_color=(255, 255, 255), padding=20):
#     """Draw text with background rectangle"""
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#     # Get text size
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
#
#     # Calculate background rectangle coordinates
#     x, y = position
#     bg_x1 = x - padding
#     bg_y1 = y - text_height - padding - baseline
#     bg_x2 = x + text_width + padding
#     bg_y2 = y + padding
#
#     # Ensure coordinates are within image bounds
#     h, w = img.shape[:2]
#     bg_x1 = max(0, bg_x1)
#     bg_y1 = max(0, bg_y1)
#     bg_x2 = min(w, bg_x2)
#     bg_y2 = min(h, bg_y2)
#
#     # Draw background rectangle
#     cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
#
#     # Draw text
#     cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)
#
#     return img
#
#
# # Load results CSV
# results = pd.read_csv('./test.csv')
#
# # Load video
# video_path = 'Sample2.mp4'
# cap = cv2.VideoCapture(video_path)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
#
# # Create a dictionary to store the best license plate crops for each unique license number
# license_plate_crops = {}
#
# # Process all detections to find the best crop for each license number
# for idx, row in results.iterrows():
#     if pd.notna(row['license_number']) and row['license_number'] != '0':
#         license_number = row['license_number']
#         score = float(row['license_number_score']) if pd.notna(row['license_number_score']) else 0
#
#         # If this is the first time seeing this license number or if this detection has a higher score
#         if license_number not in license_plate_crops or score > license_plate_crops[license_number]['score']:
#             # Get the frame for this detection
#             cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_nmr']))
#             ret, frame = cap.read()
#
#             if ret:
#                 bbox_coords = safe_bbox_parse(row['license_plate_bbox'])
#                 if bbox_coords:
#                     try:
#                         x1, y1, x2, y2 = bbox_coords
#                         license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#                         if license_crop.size > 0:
#                             # Resize crop to make it more visible
#                             crop_height = 100
#                             crop_width = int((x2 - x1) * crop_height / (y2 - y1))
#                             license_crop = cv2.resize(license_crop, (crop_width, crop_height))
#
#                             license_plate_crops[license_number] = {
#                                 'crop': license_crop,
#                                 'score': score
#                             }
#                     except Exception as e:
#                         print(f"Error processing crop for {license_number}: {e}")
#
# print(f"Processed {len(license_plate_crops)} unique license plates")
#
# # Reset video to beginning
# frame_nmr = -1
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# # Read frames and add annotations
# ret = True
# while ret:
#     ret, frame = cap.read()
#     frame_nmr += 1
#     if ret:
#         # Get all license plate detections for this frame
#         frame_detections = results[results['frame_nmr'] == frame_nmr]
#
#         for idx, row in frame_detections.iterrows():
#             try:
#                 # Parse license plate bbox
#                 lp_bbox_coords = safe_bbox_parse(row['license_plate_bbox'])
#                 if lp_bbox_coords:
#                     x1, y1, x2, y2 = lp_bbox_coords
#
#                     # Draw license plate bounding box
#                     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
#
#                     # Get license plate text
#                     license_text = str(row['license_number']) if pd.notna(row['license_number']) and row[
#                         'license_number'] != '0' else ""
#
#                     if license_text:
#                         # Position for displaying the enlarged license plate and text
#                         display_x = int(x1)
#                         display_y = int(y1) - 150  # Position above the detection
#
#                         # Ensure display position is within frame bounds
#                         if display_y < 0:
#                             display_y = int(y2) + 20  # Position below if not enough space above
#
#                         # Display the enlarged license plate crop if available
#                         if license_text in license_plate_crops:
#                             crop = license_plate_crops[license_text]['crop']
#                             crop_h, crop_w = crop.shape[:2]
#
#                             # Ensure crop fits within frame
#                             if display_x + crop_w < width and display_y + crop_h < height and display_x >= 0 and display_y >= 0:
#                                 try:
#                                     frame[display_y:display_y + crop_h, display_x:display_x + crop_w] = crop
#
#                                     # Add text below the crop
#                                     text_y = display_y + crop_h + 40
#                                     draw_text_with_background(frame, license_text, (display_x, text_y),
#                                                               font_scale=1.5, font_thickness=3,
#                                                               text_color=(0, 0, 0), bg_color=(255, 255, 255))
#                                 except Exception as e:
#                                     print(f"Error displaying crop: {e}")
#                                     # Fallback: just show text
#                                     draw_text_with_background(frame, license_text, (display_x, display_y + 30),
#                                                               font_scale=1.5, font_thickness=3,
#                                                               text_color=(0, 0, 0), bg_color=(255, 255, 255))
#                         else:
#                             # No crop available, just show text
#                             draw_text_with_background(frame, license_text, (display_x, display_y + 30),
#                                                       font_scale=1.5, font_thickness=3,
#                                                       text_color=(0, 0, 0), bg_color=(255, 255, 255))
#
#                         # Also add small text near the detection box
#                         cv2.putText(frame, license_text, (int(x1), int(y1) - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#             except Exception as e:
#                 print(f"Error processing detection in frame {frame_nmr}: {e}")
#                 continue
#
#         out.write(frame)
#
#         # Resize for display (optional - uncomment to show frame by frame)
#         # frame_display = cv2.resize(frame, (1280, 720))
#         # cv2.imshow('frame', frame_display)
#         # cv2.waitKey(1)
#
# out.release()
# cap.release()
# cv2.destroyAllWindows()
#
# # print("Video processing complete. Output saved as 'out.mp4'")
# import ast
# import cv2
# import numpy as np
# import pandas as pd
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
# results = pd.read_csv('./test_interpolated.csv')
#
# # Load video
# video_path = 'Sample4.mp4'
# cap = cv2.VideoCapture(video_path)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
#
# license_plate = {}
#
# for car_id in np.unique(results['car_id']):
#     try:
#         max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
#         best_row = results[(results['car_id'] == car_id) &
#                            (results['license_number_score'] == max_score)].iloc[0]
#         license_plate_number = best_row['license_number']
#         frame_number = best_row['frame_nmr']
#
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret, frame = cap.read()
#
#         if not ret or frame is None:
#             print(f"[Warning] Could not read frame {frame_number} for car_id {car_id}")
#             continue
#
#         # Parse and sanitize the bounding box
#         bbox_str = best_row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
#         x1, y1, x2, y2 = map(int, ast.literal_eval(bbox_str))
#
#         h, w, _ = frame.shape
#         if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
#             print(f"[Warning] Invalid bbox for car_id {car_id}: {(x1, y1, x2, y2)}")
#             continue
#
#         license_crop = frame[y1:y2, x1:x2]
#         if license_crop.size == 0:
#             print(f"[Warning] Empty crop for car_id {car_id}")
#             continue
#
#         license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
#         license_plate[car_id] = {
#             'license_crop': license_crop,
#             'license_plate_number': license_plate_number
#         }
#
#     except Exception as e:
#         print(f"[Error] Processing car_id {car_id}: {e}")
#         continue
#
#
# frame_nmr = -1
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# # Read frames
# ret = True
# while ret:
#     ret, frame = cap.read()
#     frame_nmr += 1
#
#     if not ret or frame is None:
#         break
#
#     df_ = results[results['frame_nmr'] == frame_nmr]
#
#     for row_indx in range(len(df_)):
#         try:
#             row = df_.iloc[row_indx]
#             car_x1, car_y1, car_x2, car_y2 = map(int, ast.literal_eval(
#                 row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')))
#
#             draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25,
#                         line_length_x=200, line_length_y=200)
#
#             x1, y1, x2, y2 = map(int, ast.literal_eval(
#                 row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')))
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)
#
#             car_id = row['car_id']
#             if car_id not in license_plate:
#                 continue
#
#             license_crop = license_plate[car_id]['license_crop']
#             license_text = license_plate[car_id]['license_plate_number']
#
#             H, W, _ = license_crop.shape
#             frame[car_y1 - H - 100:car_y1 - 100,
#                   int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = license_crop
#
#             # White background for text
#             frame[car_y1 - H - 400:car_y1 - H - 100,
#                   int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2)] = (255, 255, 255)
#
#             (text_width, text_height), _ = cv2.getTextSize(
#                 license_text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
#
#             cv2.putText(frame, license_text,
#                         (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
#                         cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
#
#         except Exception as e:
#             print(f"[Error] Drawing info on frame {frame_nmr}, row {row_indx}: {e}")
#             continue
#
#     out.write(frame)
#
#     # Optional: Display resized frame
#     # frame_display = cv2.resize(frame, (1280, 720))
#     # cv2.imshow('frame', frame_display)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break
#
# cap.release()
# out.release()
# # cv2.destroyAllWindows()
#

#Correct Version
# import ast
# import cv2
# import numpy as np
# import pandas as pd
#
#
# def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
#     x1, y1 = top_left
#     x2, y2 = bottom_right
#
#     cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
#     cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
#
#     cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
#     cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
#
#     cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
#     cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
#
#     cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
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
# # Load results - use interpolated data if available, otherwise original
# try:
#     results = pd.read_csv('./test_interpolated.csv')
#     print("Using interpolated data")
# except FileNotFoundError:
#     results = pd.read_csv('./test.csv')
#     print("Using original data - consider running add_missing_data.py first")
#
# # Load video
# video_path = '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/frame-190.jpg'
# cap = cv2.VideoCapture(video_path)
#
# if not cap.isOpened():
#     print("Error: Could not open video file")
#     exit()
#
# # Video writer setup
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
#
# print(f"Video info: {width}x{height} at {fps} FPS")
#
# # Extract best license plate for each car
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
#         'license_plate_number': best_row['license_number']
#     }
#
#     # Extract license plate crop from the best frame
#     cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
#     ret, frame = cap.read()
#
#     if ret:
#         # Parse bounding box coordinates
#         coords = safe_ast_eval(best_row['license_plate_bbox'])
#         x1, y1, x2, y2 = [int(c) for c in coords]
#
#         # Ensure coordinates are within frame bounds
#         x1 = max(0, min(x1, width))
#         y1 = max(0, min(y1, height))
#         x2 = max(0, min(x2, width))
#         y2 = max(0, min(y2, height))
#
#         if x2 > x1 and y2 > y1:  # Valid bounding box
#             license_crop = frame[y1:y2, x1:x2, :]
#
#             if license_crop.size > 0:
#                 # Resize license plate crop for better visibility - MUCH LARGER
#                 crop_height = 160  # Doubled height for better visibility
#                 crop_width = int((x2 - x1) * crop_height / (y2 - y1)) if (y2 - y1) > 0 else 400
#                 crop_width = max(crop_width, 300)  # Increased minimum width
#
#                 try:
#                     license_crop = cv2.resize(license_crop, (crop_width, crop_height))
#                     license_plate[car_id]['license_crop'] = license_crop
#                 except:
#                     print(f"Warning: Could not resize license crop for car {car_id}")
#
# print(f"Processed {len(license_plate)} unique cars")
#
# # Reset video to beginning
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# frame_nmr = -1
#
# # Process video frames
# ret = True
# while ret:
#     ret, frame = cap.read()
#     frame_nmr += 1
#
#     if ret:
#         # Get detections for current frame
#         df_ = results[results['frame_nmr'] == frame_nmr]
#
#         for row_idx in range(len(df_)):
#             row = df_.iloc[row_idx]
#             car_id = row['car_id']
#
#             # Parse car bounding box
#             car_coords = safe_ast_eval(row['car_bbox'])
#             car_x1, car_y1, car_x2, car_y2 = [int(c) for c in car_coords]
#
#             # Ensure car coordinates are within bounds
#             car_x1 = max(0, min(car_x1, width))
#             car_y1 = max(0, min(car_y1, height))
#             car_x2 = max(0, min(car_x2, width))
#             car_y2 = max(0, min(car_y2, height))
#
#             # Draw car bounding box
#             if car_x2 > car_x1 and car_y2 > car_y1:
#                 draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25,
#                             line_length_x=200, line_length_y=200)
#
#             # Parse license plate bounding box
#             lp_coords = safe_ast_eval(row['license_plate_bbox'])
#             lp_x1, lp_y1, lp_x2, lp_y2 = [int(c) for c in lp_coords]
#
#             # Ensure license plate coordinates are within bounds
#             lp_x1 = max(0, min(lp_x1, width))
#             lp_y1 = max(0, min(lp_y1, height))
#             lp_x2 = max(0, min(lp_x2, width))
#             lp_y2 = max(0, min(lp_y2, height))
#
#             # Draw license plate bounding box
#             if lp_x2 > lp_x1 and lp_y2 > lp_y1:
#                 cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 12)
#
#             # Add license plate crop and text if available
#             if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
#                 license_crop = license_plate[car_id]['license_crop']
#                 license_text = license_plate[car_id]['license_plate_number']
#
#                 H, W = license_crop.shape[:2]
#
#                 # Calculate position for license plate display (above the car) - MORE SPACE
#                 display_y = max(180, car_y1 - 220)  # More space from top and car
#                 center_x = (car_x1 + car_x2) // 2
#                 display_x1 = max(0, center_x - W // 2)
#                 display_x2 = min(width, display_x1 + W)
#
#                 # Adjust if the crop would go outside frame bounds
#                 if display_x2 >= width:
#                     display_x1 = width - W
#                     display_x2 = width
#                 if display_x1 < 0:
#                     display_x1 = 0
#                     display_x2 = W
#
#                 # Place license plate crop
#                 try:
#                     if (display_y + H < height and
#                             display_x1 >= 0 and display_x2 <= width and
#                             display_y >= 0):
#
#                         frame[display_y:display_y + H, display_x1:display_x2, :] = license_crop
#
#                         # Add white background for text - LARGER
#                         text_bg_y1 = max(0, display_y - 80)  # Larger text background
#                         text_bg_y2 = display_y
#                         if text_bg_y2 > text_bg_y1:
#                             frame[text_bg_y1:text_bg_y2, display_x1:display_x2, :] = (255, 255, 255)
#
#                         # Add license plate text - MUCH LARGER
#                         if license_text and license_text != '0':
#                             font_scale = 2.5  # Much larger font
#                             thickness = 6  # Thicker text
#                             (text_width, text_height), _ = cv2.getTextSize(
#                                 license_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#
#                             text_x = max(0, center_x - text_width // 2)
#                             text_y = max(text_height, display_y - 20)  # More space from license plate
#
#                             cv2.putText(frame, license_text,
#                                         (text_x, text_y),
#                                         cv2.FONT_HERSHEY_SIMPLEX,
#                                         font_scale, (0, 0, 0), thickness)
#
#                 except Exception as e:
#                     print(f"Warning: Could not place license crop for car {car_id} at frame {frame_nmr}: {e}")
#                     continue
#
#         # Write frame to output video
#         out.write(frame)
#
#         # Optional: Display progress
#         if frame_nmr % 30 == 0:
#             print(f"Processed frame {frame_nmr}")
#
# # Clean up
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#
# print(f"Video processing complete. Output saved as 'out.mp4'")
# print(f"Total frames processed: {frame_nmr + 1}")

import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #  top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #  bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #  top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #  bottom-right
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


# Load results - use interpolated data if available, otherwise original
try:
    results = pd.read_csv('./test_interpolated.csv')
    print("Using interpolated data")
except FileNotFoundError:
    results = pd.read_csv('./test.csv')
    print("Using original data - consider running add_missing_data.py first")


# Configuration - UPDATE THESE PATHS
video_path = './Sample3.mp4'  # Change this to your video path
output_path = './out.mp4'    # Change this to your desired output path

# Load video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Video info: {width}x{height} at {fps} FPS")

# Extract best license plate for each car
license_plate = {}
for car_id in results['car_id'].unique():
    car_data = results[results['car_id'] == car_id]

    # Find the frame with the highest license number score for this car
    max_score_idx = car_data['license_number_score'].astype(float).idxmax()
    best_row = results.loc[max_score_idx]

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': best_row['license_number']
    }

    # Extract license plate crop from the best frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
    ret, frame = cap.read()

    if ret:
        # Parse bounding box coordinates using safe method
        coords = safe_ast_eval(best_row['license_plate_bbox'])
        x1, y1, x2, y2 = [int(c) for c in coords]

        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 > x1 and y2 > y1:  # Valid bounding box
            license_crop = frame[y1:y2, x1:x2, :]

            if license_crop.size > 0:
                # Resize license plate crop for better visibility - MUCH LARGER
                crop_height = 160  # Doubled height for better visibility
                crop_width = int((x2 - x1) * crop_height / (y2 - y1)) if (y2 - y1) > 0 else 400
                crop_width = max(crop_width, 300)  # Increased minimum width

                try:
                    license_crop = cv2.resize(license_crop, (crop_width, crop_height))
                    license_plate[car_id]['license_crop'] = license_crop
                except:
                    print(f"Warning: Could not resize license crop for car {car_id}")

print(f"Processed {len(license_plate)} unique cars")

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_nmr = -1

# Process video frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1

    if ret:
        # Get detections for current frame
        df_ = results[results['frame_nmr'] == frame_nmr]

        for row_idx in range(len(df_)):
            row = df_.iloc[row_idx]
            car_id = row['car_id']

            # Parse car bounding box using safe method
            car_coords = safe_ast_eval(row['car_bbox'])
            car_x1, car_y1, car_x2, car_y2 = [int(c) for c in car_coords]

            # Ensure car coordinates are within bounds
            car_x1 = max(0, min(car_x1, width))
            car_y1 = max(0, min(car_y1, height))
            car_x2 = max(0, min(car_x2, width))
            car_y2 = max(0, min(car_y2, height))

            # Draw car bounding box
            if car_x2 > car_x1 and car_y2 > car_y1:
                draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

            # Parse license plate bounding box using safe method
            lp_coords = safe_ast_eval(row['license_plate_bbox'])
            lp_x1, lp_y1, lp_x2, lp_y2 = [int(c) for c in lp_coords]

            # Ensure license plate coordinates are within bounds
            lp_x1 = max(0, min(lp_x1, width))
            lp_y1 = max(0, min(lp_y1, height))
            lp_x2 = max(0, min(lp_x2, width))
            lp_y2 = max(0, min(lp_y2, height))

            # Draw license plate bounding box
            if lp_x2 > lp_x1 and lp_y2 > lp_y1:
                cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (0, 0, 255), 12)

            # Add license plate crop and text if available
            if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
                license_crop = license_plate[car_id]['license_crop']
                license_text = license_plate[car_id]['license_plate_number']

                H, W = license_crop.shape[:2]

                # Calculate position for license plate display (above the car) - MORE SPACE
                display_y = max(180, car_y1 - 220)  # More space from top and car
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

                        frame[display_y:display_y + H, display_x1:display_x2, :] = license_crop

                        # Add white background for text - LARGER
                        text_bg_y1 = max(0, display_y - 80)  # Larger text background
                        text_bg_y2 = display_y
                        if text_bg_y2 > text_bg_y1:
                            frame[text_bg_y1:text_bg_y2, display_x1:display_x2, :] = (255, 255, 255)

                        # Add license plate text - MUCH LARGER
                        if license_text and license_text != '0':
                            font_scale = 2.5  # Much larger font
                            thickness = 6  # Thicker text
                            (text_width, text_height), _ = cv2.getTextSize(
                                license_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                            text_x = max(0, center_x - text_width // 2)
                            text_y = max(text_height, display_y - 20)  # More space from license plate

                            cv2.putText(frame, license_text,
                                        (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, (0, 0, 0), thickness)

                except Exception as e:
                    print(f"Warning: Could not place license crop for car {car_id} at frame {frame_nmr}: {e}")
                    continue

        # Write frame to output video
        out.write(frame)

        # Optional: Display progress
        if frame_nmr % 30 == 0:
            print(f"Processed frame {frame_nmr}")

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved as '{output_path}'")
print(f"Total frames processed: {frame_nmr + 1}")
