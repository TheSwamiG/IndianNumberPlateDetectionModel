# import csv
# import numpy as np
# from scipy.interpolate import interp1d
#
#
# def interpolate_bounding_boxes(data):
#     # Extract necessary data columns from input data
#     frame_numbers = np.array([int(row['frame_nmr']) for row in data])
#     car_ids = np.array([int(float(row['car_id'])) for row in data])
#     car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
#     license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])
#
#     interpolated_data = []
#     unique_car_ids = np.unique(car_ids)
#     for car_id in unique_car_ids:
#
#         frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
#         print(frame_numbers_, car_id)
#
#         # Filter data for a specific car ID
#         car_mask = car_ids == car_id
#         car_frame_numbers = frame_numbers[car_mask]
#         car_bboxes_interpolated = []
#         license_plate_bboxes_interpolated = []
#
#         first_frame_number = car_frame_numbers[0]
#         last_frame_number = car_frame_numbers[-1]
#
#         for i in range(len(car_bboxes[car_mask])):
#             frame_number = car_frame_numbers[i]
#             car_bbox = car_bboxes[car_mask][i]
#             license_plate_bbox = license_plate_bboxes[car_mask][i]
#
#             if i > 0:
#                 prev_frame_number = car_frame_numbers[i-1]
#                 prev_car_bbox = car_bboxes_interpolated[-1]
#                 prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
#
#                 if frame_number - prev_frame_number > 1:
#                     # Interpolate missing frames' bounding boxes
#                     frames_gap = frame_number - prev_frame_number
#                     x = np.array([prev_frame_number, frame_number])
#                     x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
#                     interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
#                     interpolated_car_bboxes = interp_func(x_new)
#                     interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
#                     interpolated_license_plate_bboxes = interp_func(x_new)
#
#                     car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
#                     license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
#
#             car_bboxes_interpolated.append(car_bbox)
#             license_plate_bboxes_interpolated.append(license_plate_bbox)
#
#         for i in range(len(car_bboxes_interpolated)):
#             frame_number = first_frame_number + i
#             row = {}
#             row['frame_nmr'] = str(frame_number)
#             row['car_id'] = str(car_id)
#             row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
#             row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))
#
#             if str(frame_number) not in frame_numbers_:
#                 # Imputed row, set the following fields to '0'
#                 row['license_plate_bbox_score'] = '0'
#                 row['license_number'] = '0'
#                 row['license_number_score'] = '0'
#             else:
#                 # Original row, retrieve values from the input data if available
#                 original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
#                 row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
#                 row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
#                 row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'
#
#             interpolated_data.append(row)
#
#     return interpolated_data
#
#
# # Load the CSV file
# with open('test.csv', 'r') as file:
#     reader = csv.DictReader(file)
#     data = list(reader)
#
# # Interpolate missing data
# interpolated_data = interpolate_bounding_boxes(data)
#
# # Write updated data to a new CSV file
# header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
# with open('test_interpolated.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()
#     writer.writerows(interpolated_data)
# import csv
# import numpy as np
# from scipy.interpolate import interp1d
#
#
# def interpolate_bounding_boxes(data):
#     # Separate vehicle data from standalone license plates
#     vehicle_data = []
#     standalone_data = []
#
#     for row in data:
#         try:
#             # Check if car_id is numeric (tracked vehicle)
#             int(float(row['car_id']))
#             # Also check if car_bbox is not 'N/A' (has associated vehicle)
#             if row['car_bbox'] != 'N/A':
#                 vehicle_data.append(row)
#             else:
#                 standalone_data.append(row)
#         except ValueError:
#             # This is a standalone license plate (string ID like 'LP_18_235_1785')
#             standalone_data.append(row)
#
#     if not vehicle_data:
#         print("No tracked vehicles found for interpolation")
#         return data
#
#     print(f"Processing {len(vehicle_data)} vehicle entries and {len(standalone_data)} standalone license plates")
#
#     # Extract necessary data columns from vehicle data only
#     frame_numbers = np.array([int(row['frame_nmr']) for row in vehicle_data])
#     car_ids = np.array([int(float(row['car_id'])) for row in vehicle_data])
#     car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in vehicle_data])
#     license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in vehicle_data])
#
#     interpolated_data = []
#     unique_car_ids = np.unique(car_ids)
#     for car_id in unique_car_ids:
#
#         frame_numbers_ = [p['frame_nmr'] for p in vehicle_data if int(float(p['car_id'])) == int(float(car_id))]
#         print(frame_numbers_, car_id)
#
#         # Filter data for a specific car ID
#         car_mask = car_ids == car_id
#         car_frame_numbers = frame_numbers[car_mask]
#         car_bboxes_interpolated = []
#         license_plate_bboxes_interpolated = []
#
#         first_frame_number = car_frame_numbers[0]
#         last_frame_number = car_frame_numbers[-1]
#
#         for i in range(len(car_bboxes[car_mask])):
#             frame_number = car_frame_numbers[i]
#             car_bbox = car_bboxes[car_mask][i]
#             license_plate_bbox = license_plate_bboxes[car_mask][i]
#
#             if i > 0:
#                 prev_frame_number = car_frame_numbers[i - 1]
#                 prev_car_bbox = car_bboxes_interpolated[-1]
#                 prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
#
#                 if frame_number - prev_frame_number > 1:
#                     # Interpolate missing frames' bounding boxes
#                     frames_gap = frame_number - prev_frame_number
#                     x = np.array([prev_frame_number, frame_number])
#                     x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
#                     interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
#                     interpolated_car_bboxes = interp_func(x_new)
#                     interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0,
#                                            kind='linear')
#                     interpolated_license_plate_bboxes = interp_func(x_new)
#
#                     car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
#                     license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
#
#             car_bboxes_interpolated.append(car_bbox)
#             license_plate_bboxes_interpolated.append(license_plate_bbox)
#
#         for i in range(len(car_bboxes_interpolated)):
#             frame_number = first_frame_number + i
#             row = {}
#             row['frame_nmr'] = str(frame_number)
#             row['car_id'] = str(car_id)
#             row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
#             row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))
#
#             if str(frame_number) not in frame_numbers_:
#                 # Imputed row, set the following fields to '0'
#                 row['license_plate_bbox_score'] = '0'
#                 row['license_number'] = '0'
#                 row['license_number_score'] = '0'
#             else:
#                 # Original row, retrieve values from the input data if available
#                 original_row = [p for p in vehicle_data if
#                                 int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][
#                     0]
#                 row['license_plate_bbox_score'] = original_row[
#                     'license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
#                 row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
#                 row['license_number_score'] = original_row[
#                     'license_number_score'] if 'license_number_score' in original_row else '0'
#
#             interpolated_data.append(row)
#
#     # Add standalone license plates to the interpolated data (no changes needed)
#     interpolated_data.extend(standalone_data)
#
#     return interpolated_data
#
#
# # Load the CSV file
# with open('test.csv', 'r') as file:
#     reader = csv.DictReader(file)
#     data = list(reader)
#
# # Interpolate missing data
# interpolated_data = interpolate_bounding_boxes(data)
#
# # Write updated data to a new CSV file
# header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
#           'license_number_score']
# with open('test_interpolated.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()
#     writer.writerows(interpolated_data)
# import csv
# import pandas as pd
# from collections import defaultdict
#
#
# def process_license_plate_data(data):
#     """
#     Process license plate data to remove duplicates and keep the best detections.
#     Since we're not tracking vehicles, we'll group by license plate text and
#     keep the detection with the highest confidence score.
#     """
#
#     # First, let's check what columns are available
#     if data:
#         print("Available columns:", list(data[0].keys()))
#
#     # Group detections by license plate text
#     license_groups = defaultdict(list)
#
#     for row in data:
#         license_text = row.get('license_number', '')
#         if license_text and license_text != '0':
#             license_groups[license_text].append(row)
#
#     processed_data = []
#
#     # For each unique license plate, keep the best detection(s)
#     for license_text, detections in license_groups.items():
#         if len(detections) == 1:
#             # Only one detection, keep it
#             processed_data.extend(detections)
#         else:
#             # Multiple detections, sort by score and keep the best ones
#             # Check which score column exists
#             score_column = None
#             if 'license_number_score' in detections[0]:
#                 score_column = 'license_number_score'
#             elif 'license_plate_bbox_score' in detections[0]:
#                 score_column = 'license_plate_bbox_score'
#             elif 'score' in detections[0]:
#                 score_column = 'score'
#
#             if score_column:
#                 try:
#                     detections.sort(key=lambda x: float(x[score_column]) if x[score_column] else 0, reverse=True)
#                 except (ValueError, KeyError):
#                     # If sorting fails, just keep original order
#                     pass
#
#             # Keep top detections (you can adjust this number)
#             max_detections_per_plate = 3
#             processed_data.extend(detections[:max_detections_per_plate])
#
#     # Also add any detections that couldn't be read (license_number is '0' or empty)
#     for row in data:
#         license_text = row.get('license_number', '')
#         if not license_text or license_text == '0':
#             processed_data.append(row)
#
#     # Sort by frame number if frame_nmr exists
#     try:
#         processed_data.sort(key=lambda x: int(x.get('frame_nmr', 0)) if x.get('frame_nmr') else 0)
#     except (ValueError, KeyError):
#         # If sorting by frame fails, keep original order
#         pass
#
#     return processed_data
#
#
# # Load the CSV file
# try:
#     with open('test.csv', 'r') as file:
#         reader = csv.DictReader(file)
#         data = list(reader)
# except FileNotFoundError:
#     print("Error: 'test.csv' file not found!")
#     exit(1)
#
# print(f"Original data: {len(data)} detections")
#
# # Process the data
# processed_data = process_license_plate_data(data)
#
# print(f"Processed data: {len(processed_data)} detections")
#
# # Determine the actual header from the data
# if processed_data:
#     actual_header = list(processed_data[0].keys())
# else:
#     # Fallback header
#     actual_header = ['frame_nmr', 'plate_id', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number']
#
# # Write processed data to a new CSV file
# with open('test_processed.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=actual_header)
#     writer.writeheader()
#     writer.writerows(processed_data)
#
# print("Processed data saved to 'test_processed.csv'")
#
# # Create a summary report
# unique_plates = set()
# for row in processed_data:
#     license_text = row.get('license_number', '')
#     if license_text and license_text != '0':
#         unique_plates.add(license_text)
#
# print(f"\nSummary:")
# print(f"Total unique license plates detected: {len(unique_plates)}")
# if unique_plates:
#     print(f"License plates found: {sorted(list(unique_plates))}")
# else:
#     print("No valid license plates found in the data.")

# Correct Version
# import csv
# import numpy as np
# from scipy.interpolate import interp1d
#
#
# def interpolate_bounding_boxes(data):
#     # Extract necessary data columns from input data
#     frame_numbers = np.array([int(row['frame_nmr']) for row in data])
#     car_ids = np.array([int(float(row['car_id'])) for row in data])
#     car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
#     license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])
#
#     interpolated_data = []
#     unique_car_ids = np.unique(car_ids)
#     for car_id in unique_car_ids:
#
#         frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
#         print(frame_numbers_, car_id)
#
#         # Filter data for a specific car ID
#         car_mask = car_ids == car_id
#         car_frame_numbers = frame_numbers[car_mask]
#         car_bboxes_interpolated = []
#         license_plate_bboxes_interpolated = []
#
#         first_frame_number = car_frame_numbers[0]
#         last_frame_number = car_frame_numbers[-1]
#
#         for i in range(len(car_bboxes[car_mask])):
#             frame_number = car_frame_numbers[i]
#             car_bbox = car_bboxes[car_mask][i]
#             license_plate_bbox = license_plate_bboxes[car_mask][i]
#
#             if i > 0:
#                 prev_frame_number = car_frame_numbers[i-1]
#                 prev_car_bbox = car_bboxes_interpolated[-1]
#                 prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
#
#                 if frame_number - prev_frame_number > 1:
#                     # Interpolate missing frames' bounding boxes
#                     frames_gap = frame_number - prev_frame_number
#                     x = np.array([prev_frame_number, frame_number])
#                     x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
#                     interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
#                     interpolated_car_bboxes = interp_func(x_new)
#                     interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
#                     interpolated_license_plate_bboxes = interp_func(x_new)
#
#                     car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
#                     license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
#
#             car_bboxes_interpolated.append(car_bbox)
#             license_plate_bboxes_interpolated.append(license_plate_bbox)
#
#         for i in range(len(car_bboxes_interpolated)):
#             frame_number = first_frame_number + i
#             row = {}
#             row['frame_nmr'] = str(frame_number)
#             row['car_id'] = str(car_id)
#             row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
#             row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))
#
#             if str(frame_number) not in frame_numbers_:
#                 # Imputed row, set the following fields to '0'
#                 row['license_plate_bbox_score'] = '0'
#                 row['license_number'] = '0'
#                 row['license_number_score'] = '0'
#             else:
#                 # Original row, retrieve values from the input data if available
#                 original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
#                 row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
#                 row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
#                 row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'
#
#             interpolated_data.append(row)
#
#     return interpolated_data
#
#
# # Load the CSV file
# with open('test.csv', 'r') as file:
#     reader = csv.DictReader(file)
#     data = list(reader)
#
# # Interpolate missing data
# interpolated_data = interpolate_bounding_boxes(data)
#
# # Write updated data to a new CSV file
# header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
# with open('test_interpolated.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()
#     writer.writerows(interpolated_data)

#Correct Version
# import csv
# import numpy as np
# from scipy.interpolate import interp1d
#
#
# def interpolate_bounding_boxes(data):
#     # Extract necessary data columns from input data
#     frame_numbers = np.array([int(row['frame_nmr']) for row in data])
#     car_ids = np.array([int(float(row['car_id'])) for row in data])
#
#     # Fix bounding box parsing - handle both formats
#     car_bboxes = []
#     license_plate_bboxes = []
#
#     for row in data:
#         # Clean and parse car bbox
#         car_bbox_str = row['car_bbox'].replace('[', '').replace(']', '').replace(',', ' ')
#         car_bbox_str = ' '.join(car_bbox_str.split())  # normalize whitespace
#         car_bbox = list(map(float, car_bbox_str.split()))
#         car_bboxes.append(car_bbox)
#
#         # Clean and parse license plate bbox
#         lp_bbox_str = row['license_plate_bbox'].replace('[', '').replace(']', '').replace(',', ' ')
#         lp_bbox_str = ' '.join(lp_bbox_str.split())  # normalize whitespace
#         lp_bbox = list(map(float, lp_bbox_str.split()))
#         license_plate_bboxes.append(lp_bbox)
#
#     car_bboxes = np.array(car_bboxes)
#     license_plate_bboxes = np.array(license_plate_bboxes)
#
#     interpolated_data = []
#     unique_car_ids = np.unique(car_ids)
#
#     for car_id in unique_car_ids:
#         print(f"Processing car_id: {car_id}")
#
#         # Filter data for a specific car ID
#         car_mask = car_ids == car_id
#         car_frame_numbers = frame_numbers[car_mask]
#         car_bboxes_filtered = car_bboxes[car_mask]
#         license_plate_bboxes_filtered = license_plate_bboxes[car_mask]
#
#         # Sort by frame number to ensure proper ordering
#         sort_indices = np.argsort(car_frame_numbers)
#         car_frame_numbers = car_frame_numbers[sort_indices]
#         car_bboxes_filtered = car_bboxes_filtered[sort_indices]
#         license_plate_bboxes_filtered = license_plate_bboxes_filtered[sort_indices]
#
#         # Get original data for this car
#         original_data = [row for row in data if int(float(row['car_id'])) == car_id]
#         original_data.sort(key=lambda x: int(x['frame_nmr']))
#
#         first_frame = car_frame_numbers[0]
#         last_frame = car_frame_numbers[-1]
#
#         # Create interpolated data for all frames between first and last
#         all_frames = np.arange(first_frame, last_frame + 1)
#
#         # Interpolate car bboxes
#         if len(car_frame_numbers) > 1:
#             interp_car_x1 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 0], kind='linear',
#                                      fill_value='extrapolate')
#             interp_car_y1 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 1], kind='linear',
#                                      fill_value='extrapolate')
#             interp_car_x2 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 2], kind='linear',
#                                      fill_value='extrapolate')
#             interp_car_y2 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 3], kind='linear',
#                                      fill_value='extrapolate')
#
#             interp_lp_x1 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 0], kind='linear',
#                                     fill_value='extrapolate')
#             interp_lp_y1 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 1], kind='linear',
#                                     fill_value='extrapolate')
#             interp_lp_x2 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 2], kind='linear',
#                                     fill_value='extrapolate')
#             interp_lp_y2 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 3], kind='linear',
#                                     fill_value='extrapolate')
#         else:
#             # Only one frame, use constant values
#             interp_car_x1 = lambda x: car_bboxes_filtered[0, 0]
#             interp_car_y1 = lambda x: car_bboxes_filtered[0, 1]
#             interp_car_x2 = lambda x: car_bboxes_filtered[0, 2]
#             interp_car_y2 = lambda x: car_bboxes_filtered[0, 3]
#
#             interp_lp_x1 = lambda x: license_plate_bboxes_filtered[0, 0]
#             interp_lp_y1 = lambda x: license_plate_bboxes_filtered[0, 1]
#             interp_lp_x2 = lambda x: license_plate_bboxes_filtered[0, 2]
#             interp_lp_y2 = lambda x: license_plate_bboxes_filtered[0, 3]
#
#         for frame_num in all_frames:
#             row = {}
#             row['frame_nmr'] = str(frame_num)
#             row['car_id'] = str(car_id)
#             row[
#                 'car_bbox'] = f"[{interp_car_x1(frame_num):.2f} {interp_car_y1(frame_num):.2f} {interp_car_x2(frame_num):.2f} {interp_car_y2(frame_num):.2f}]"
#             row[
#                 'license_plate_bbox'] = f"[{interp_lp_x1(frame_num):.2f} {interp_lp_y1(frame_num):.2f} {interp_lp_x2(frame_num):.2f} {interp_lp_y2(frame_num):.2f}]"
#
#             # Check if this frame exists in original data
#             original_frame_data = [d for d in original_data if int(d['frame_nmr']) == frame_num]
#
#             if original_frame_data:
#                 # Original frame - use original values for scores and license number
#                 orig = original_frame_data[0]
#                 row['license_plate_bbox_score'] = orig.get('license_plate_bbox_score', '0')
#                 row['license_number'] = orig.get('license_number', '0')
#                 row['license_number_score'] = orig.get('license_number_score', '0')
#             else:
#                 # Interpolated frame - set scores to 0
#                 row['license_plate_bbox_score'] = '0'
#                 row['license_number'] = '0'
#                 row['license_number_score'] = '0'
#
#             interpolated_data.append(row)
#
#     return interpolated_data
#
#
# # Load the CSV file
# with open('test.csv', 'r') as file:
#     reader = csv.DictReader(file)
#     data = list(reader)
#
# # Interpolate missing data
# interpolated_data = interpolate_bounding_boxes(data)
#
# # Write updated data to a new CSV file
# header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
#           'license_number_score']
# with open('test_interpolated.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()
#     writer.writerows(interpolated_data)
#
# print(f"Interpolated data saved to test_interpolated.csv with {len(interpolated_data)} rows")

import csv
import numpy as np
from scipy.interpolate import interp1d
import ast


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


def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])

    # Use the safe parsing function for bounding boxes
    car_bboxes = []
    license_plate_bboxes = []

    for row in data:
        # Parse car bbox using safe method
        car_bbox = safe_ast_eval(row['car_bbox'])
        car_bboxes.append(car_bbox)

        # Parse license plate bbox using safe method
        lp_bbox = safe_ast_eval(row['license_plate_bbox'])
        license_plate_bboxes.append(lp_bbox)

    car_bboxes = np.array(car_bboxes)
    license_plate_bboxes = np.array(license_plate_bboxes)

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        print(f"Processing car_id: {car_id}")

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_filtered = car_bboxes[car_mask]
        license_plate_bboxes_filtered = license_plate_bboxes[car_mask]

        # Sort by frame number to ensure proper ordering
        sort_indices = np.argsort(car_frame_numbers)
        car_frame_numbers = car_frame_numbers[sort_indices]
        car_bboxes_filtered = car_bboxes_filtered[sort_indices]
        license_plate_bboxes_filtered = license_plate_bboxes_filtered[sort_indices]

        # Get original data for this car
        original_data = [row for row in data if int(float(row['car_id'])) == car_id]
        original_data.sort(key=lambda x: int(x['frame_nmr']))

        first_frame = car_frame_numbers[0]
        last_frame = car_frame_numbers[-1]

        # Create interpolated data for all frames between first and last
        all_frames = np.arange(first_frame, last_frame + 1)

        # Interpolate car bboxes
        if len(car_frame_numbers) > 1:
            interp_car_x1 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 0], kind='linear',
                                     fill_value='extrapolate')
            interp_car_y1 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 1], kind='linear',
                                     fill_value='extrapolate')
            interp_car_x2 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 2], kind='linear',
                                     fill_value='extrapolate')
            interp_car_y2 = interp1d(car_frame_numbers, car_bboxes_filtered[:, 3], kind='linear',
                                     fill_value='extrapolate')

            interp_lp_x1 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 0], kind='linear',
                                    fill_value='extrapolate')
            interp_lp_y1 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 1], kind='linear',
                                    fill_value='extrapolate')
            interp_lp_x2 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 2], kind='linear',
                                    fill_value='extrapolate')
            interp_lp_y2 = interp1d(car_frame_numbers, license_plate_bboxes_filtered[:, 3], kind='linear',
                                    fill_value='extrapolate')
        else:
            # Only one frame, use constant values
            interp_car_x1 = lambda x: car_bboxes_filtered[0, 0]
            interp_car_y1 = lambda x: car_bboxes_filtered[0, 1]
            interp_car_x2 = lambda x: car_bboxes_filtered[0, 2]
            interp_car_y2 = lambda x: car_bboxes_filtered[0, 3]

            interp_lp_x1 = lambda x: license_plate_bboxes_filtered[0, 0]
            interp_lp_y1 = lambda x: license_plate_bboxes_filtered[0, 1]
            interp_lp_x2 = lambda x: license_plate_bboxes_filtered[0, 2]
            interp_lp_y2 = lambda x: license_plate_bboxes_filtered[0, 3]

        for frame_num in all_frames:
            row = {}
            row['frame_nmr'] = str(frame_num)
            row['car_id'] = str(car_id)
            row[
                'car_bbox'] = f"[{interp_car_x1(frame_num):.2f} {interp_car_y1(frame_num):.2f} {interp_car_x2(frame_num):.2f} {interp_car_y2(frame_num):.2f}]"
            row[
                'license_plate_bbox'] = f"[{interp_lp_x1(frame_num):.2f} {interp_lp_y1(frame_num):.2f} {interp_lp_x2(frame_num):.2f} {interp_lp_y2(frame_num):.2f}]"

            # Check if this frame exists in original data
            original_frame_data = [d for d in original_data if int(d['frame_nmr']) == frame_num]

            if original_frame_data:
                # Original frame - use original values for scores and license number
                orig = original_frame_data[0]
                row['license_plate_bbox_score'] = orig.get('license_plate_bbox_score', '0')
                row['license_number'] = orig.get('license_number', '0')
                row['license_number_score'] = orig.get('license_number_score', '0')
            else:
                # Interpolated frame - set scores to 0
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'

            interpolated_data.append(row)

    return interpolated_data


# Load the CSV file
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
          'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

print(f"Interpolated data saved to test_interpolated.csv with {len(interpolated_data)} rows")