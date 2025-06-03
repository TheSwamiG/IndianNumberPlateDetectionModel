# import string
# import easyocr
#
# # Initialize the OCR reader
# reader = easyocr.Reader(['en'], gpu=False)
#
# # Mapping dictionaries for character conversion
# dict_char_to_int = {'O': '0',
#                     'I': '1',
#                     'J': '3',
#                     'A': '4',
#                     'G': '6',
#                     'S': '5'}
#
# dict_int_to_char = {'0': 'O',
#                     '1': 'I',
#                     '3': 'J',
#                     '4': 'A',
#                     '6': 'G',
#                     '5': 'S'}
#
#
# def write_csv(results, output_path):
#     """
#     Write the results to a CSV file.
#
#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output CSV file.
#     """
#     with open(output_path, 'w') as f:
#         f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
#                                                 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
#                                                 'license_number_score'))
#
#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 print(results[frame_nmr][car_id])
#                 if 'car' in results[frame_nmr][car_id].keys() and \
#                    'license_plate' in results[frame_nmr][car_id].keys() and \
#                    'text' in results[frame_nmr][car_id]['license_plate'].keys():
#                     f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
#                                                             car_id,
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['car']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][3]),
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][3]),
#                                                             results[frame_nmr][car_id]['license_plate']['bbox_score'],
#                                                             results[frame_nmr][car_id]['license_plate']['text'],
#                                                             results[frame_nmr][car_id]['license_plate']['text_score'])
#                             )
#         f.close()
#
#
# def license_complies_format(text):
#     """
#     Check if the license plate text complies with the required format.
#
#     Args:
#         text (str): License plate text.
#
#     Returns:
#         bool: True if the license plate complies with the format, False otherwise.
#     """
#     if len(text) != 7:
#         return False
#
#     if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
#        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
#        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
#        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
#        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
#        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
#        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
#         return True
#     else:
#         return False
#
#
# def format_license(text):
#     """
#     Format the license plate text by converting characters using the mapping dictionaries.
#
#     Args:
#         text (str): License plate text.
#
#     Returns:
#         str: Formatted license plate text.
#     """
#     license_plate_ = ''
#     mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
#                2: dict_char_to_int, 3: dict_char_to_int}
#     for j in [0, 1, 2, 3, 4, 5, 6]:
#         if text[j] in mapping[j].keys():
#             license_plate_ += mapping[j][text[j]]
#         else:
#             license_plate_ += text[j]
#
#     return license_plate_
#
#
# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image.
#
#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
#
#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score.
#     """
#
#     detections = reader.readtext(license_plate_crop)
#
#     for detection in detections:
#         bbox, text, score = detection
#
#         text = text.upper().replace(' ', '')
#
#         if license_complies_format(text):
#             return format_license(text), score
#
#     return None, None
#
#
# def get_car(license_plate, vehicle_track_ids):
#     """
#     Retrieve the vehicle coordinates and ID based on the license plate coordinates.
#
#     Args:
#         license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
#         vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
#
#     Returns:
#         tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
#     """
#     x1, y1, x2, y2, score, class_id = license_plate
#
#     foundIt = False
#     for j in range(len(vehicle_track_ids)):
#         xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
#
#         if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
#             car_indx = j
#             foundIt = True
#             break
#
#     if foundIt:
#         return vehicle_track_ids[car_indx]
#
#     return -1, -1, -1, -1, -1
#

# import string
# import easyocr
# import re
#
# # Initialize the OCR reader
# reader = easyocr.Reader(['en'], gpu=False)
#
# # Mapping dictionaries for character conversion
# dict_char_to_int = {'O': '0',
#                     'I': '1',
#                     'J': '3',
#                     'A': '4',
#                     'G': '6',
#                     'S': '5',
#                     'Z': '2',
#                     'T': '7',
#                     'B': '8'}
#
# dict_int_to_char = {'0': 'O',
#                     '1': 'I',
#                     '3': 'J',
#                     '4': 'A',
#                     '6': 'G',
#                     '5': 'S',
#                     '2': 'Z',
#                     '7': 'T',
#                     '8': 'B'}
#
# # Indian state/UT codes for validation
# INDIAN_STATE_CODES = [
#     'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA',
#     'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 'RJ', 'SK', 'TN',
#     'TS', 'TR', 'UK', 'UP', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'LD', 'PY'
# ]
#
#
# def write_csv(results, output_path):
#     """
#     Write the results to a CSV file for license plates only.
#
#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output CSV file.
#     """
#     with open(output_path, 'w') as f:
#         f.write('{},{},{},{},{},{}\n'.format('frame_nmr', 'plate_id', 'license_plate_bbox',
#                                          'license_plate_bbox_score', 'license_number', 'license_number_score'))
#
#         for frame_nmr in results.keys():
#             for plate_id in results[frame_nmr].keys():
#                 if 'license_plate' in results[frame_nmr][plate_id].keys() and \
#                         'text' in results[frame_nmr][plate_id]['license_plate'].keys():
#
#                     f.write('{},{},{},{},{},{}\n'.format(frame_nmr,
#                                                         plate_id,
#                                                         '[{} {} {} {}]'.format(
#                                                             results[frame_nmr][plate_id]['license_plate']['bbox'][0],
#                                                             results[frame_nmr][plate_id]['license_plate']['bbox'][1],
#                                                             results[frame_nmr][plate_id]['license_plate']['bbox'][2],
#                                                             results[frame_nmr][plate_id]['license_plate']['bbox'][3]),
#                                                         results[frame_nmr][plate_id]['license_plate']['bbox_score'],
#                                                         results[frame_nmr][plate_id]['license_plate']['text'],
#                                                         results[frame_nmr][plate_id]['license_plate']['text_score'])
#                             )
#         f.close()
#
#
# def license_complies_format(text):
#     """
#     Check if the license plate text complies with Indian format.
#     Indian format: XX## XX #### (e.g., MH12 AB 1234)
#
#     Args:
#         text (str): License plate text.
#
#     Returns:
#         bool: True if the license plate complies with the format, False otherwise.
#     """
#     # Remove spaces and convert to uppercase
#     text = text.replace(' ', '').upper()
#
#     # Check length (should be 10 characters without spaces)
#     if len(text) != 10:
#         return False
#
#     # Pattern: XX##XX####
#     # Positions 0,1: State code (letters)
#     # Positions 2,3: District code (numbers)
#     # Positions 4,5: Series code (letters)
#     # Positions 6,7,8,9: Registration number (numbers)
#
#     if not ((text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
#             (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
#             (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and
#             (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and
#             (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and
#             (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and
#             (text[6] in string.digits or text[6] in dict_char_to_int.keys()) and
#             (text[7] in string.digits or text[7] in dict_char_to_int.keys()) and
#             (text[8] in string.digits or text[8] in dict_char_to_int.keys()) and
#             (text[9] in string.digits or text[9] in dict_char_to_int.keys())):
#         return False
#
#     return True
#
#
# def validate_indian_state_code(state_code):
#     """
#     Validate if the state code is a valid Indian state/UT code.
#
#     Args:
#         state_code (str): Two-letter state code
#
#     Returns:
#         bool: True if valid state code, False otherwise
#     """
#     return state_code.upper() in INDIAN_STATE_CODES
#
#
# def format_license(text):
#     """
#     Format the license plate text by converting characters using the mapping dictionaries.
#     Applies specific mapping for each position based on Indian license plate format.
#
#     Args:
#         text (str): License plate text (without spaces).
#
#     Returns:
#         str: Formatted license plate text with proper spacing.
#     """
#     text = text.replace(' ', '').upper()
#     license_plate_ = ''
#
#     # Mapping for each position in Indian format XX##XX####
#     # Positions 0,1,4,5: Should be letters
#     # Positions 2,3,6,7,8,9: Should be numbers
#     mapping = {
#         0: dict_int_to_char,  # State code - letter
#         1: dict_int_to_char,  # State code - letter
#         2: dict_char_to_int,  # District code - number
#         3: dict_char_to_int,  # District code - number
#         4: dict_int_to_char,  # Series code - letter
#         5: dict_int_to_char,  # Series code - letter
#         6: dict_char_to_int,  # Registration - number
#         7: dict_char_to_int,  # Registration - number
#         8: dict_char_to_int,  # Registration - number
#         9: dict_char_to_int  # Registration - number
#     }
#
#     for i in range(10):
#         if text[i] in mapping[i].keys():
#             license_plate_ += mapping[i][text[i]]
#         else:
#             license_plate_ += text[i]
#
#     # Format with spaces: XX## XX ####
#     formatted = f"{license_plate_[:2]}{license_plate_[2:4]} {license_plate_[4:6]} {license_plate_[6:]}"
#     return formatted
#
#
# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image.
#
#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
#
#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score.
#     """
#     detections = reader.readtext(license_plate_crop)
#
#     for detection in detections:
#         bbox, text, score = detection
#
#         # Clean the text
#         text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
#
#         # Try different variations to handle OCR errors
#         text_variations = [text]
#
#         # Add variation with common OCR substitutions
#         text_corrected = text
#         for char, replacement in dict_char_to_int.items():
#             text_corrected = text_corrected.replace(char, replacement)
#         for char, replacement in dict_int_to_char.items():
#             text_corrected = text_corrected.replace(char, replacement)
#         text_variations.append(text_corrected)
#
#         # Check each variation
#         for text_var in text_variations:
#             if license_complies_format(text_var):
#                 formatted_plate = format_license(text_var)
#
#                 # Additional validation: check if state code is valid
#                 state_code = formatted_plate[:2]
#                 if validate_indian_state_code(state_code):
#                     return formatted_plate, score
#                 else:
#                     # If state code is invalid, still return it but with lower confidence
#                     return formatted_plate, score * 0.8
#
#     return None, None

# Correct Version
# import string
# import easyocr
#
#
# import re
#
# # Initialize the OCR reader
# reader = easyocr.Reader(['en'], gpu=False)
#
#
# # Mapping dictionaries for character conversion
# dict_char_to_int = {'O': '0',
#                     'I': '1',
#                     'J': '3',
#                     'A': '4',
#                     'G': '6',
#                     'S': '5',
#                     'Z': '2',
#                     'T': '7',
#                     'B': '8'}
#
# dict_int_to_char = {'0': 'O',
#                     '1': 'I',
#                     '3': 'J',
#                     '4': 'A',
#                     '6': 'G',
#                     '5': 'S',
#                     '2': 'Z',
#                     '7': 'T',
#                     '8': 'B'}
#
# # Indian state/UT codes for validation
# INDIAN_STATE_CODES = [
#     'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA',
#     'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 'RJ', 'SK', 'TN',
#     'TS', 'TR', 'UK', 'UP', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'LD', 'PY'
# ]
#
#
# def write_csv(results, output_path):
#     """
#     Write the results to a CSV file.
#
#     Args:
#         results (dict): Dictionary containing the results.
#         output_path (str): Path to the output CSV file.
#     """
#     with open(output_path, 'w') as f:
#         f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
#                                                 'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
#                                                 'license_number_score'))
#
#         for frame_nmr in results.keys():
#             for car_id in results[frame_nmr].keys():
#                 print(results[frame_nmr][car_id])
#                 if 'car' in results[frame_nmr][car_id].keys() and \
#                         'license_plate' in results[frame_nmr][car_id].keys() and \
#                         'text' in results[frame_nmr][car_id]['license_plate'].keys():
#                     f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
#                                                             car_id,
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['car']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['car']['bbox'][3]),
#                                                             '[{} {} {} {}]'.format(
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][0],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][1],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][2],
#                                                                 results[frame_nmr][car_id]['license_plate']['bbox'][3]),
#                                                             results[frame_nmr][car_id]['license_plate']['bbox_score'],
#                                                             results[frame_nmr][car_id]['license_plate']['text'],
#                                                             results[frame_nmr][car_id]['license_plate']['text_score'])
#                             )
#         f.close()
#
#
# def license_complies_format(text):
#     """
#     Check if the license plate text complies with Indian format.
#     Indian format: XX## XX #### (e.g., MH12 AB 1234)
#
#     Args:
#         text (str): License plate text.
#
#     Returns:
#         bool: True if the license plate complies with the format, False otherwise.
#     """
#     # Remove spaces and convert to uppercase
#     text = text.replace(' ', '').upper()
#
#     # Check length (should be 10 characters without spaces)
#     if len(text) != 10:
#         return False
#
#     # Pattern: XX##XX####
#     # Positions 0,1: State code (letters)
#     # Positions 2,3: District code (numbers)
#     # Positions 4,5: Series code (letters)
#     # Positions 6,7,8,9: Registration number (numbers)
#
#     if not ((text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
#             (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
#             (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and
#             (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and
#             (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and
#             (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and
#             (text[6] in string.digits or text[6] in dict_char_to_int.keys()) and
#             (text[7] in string.digits or text[7] in dict_char_to_int.keys()) and
#             (text[8] in string.digits or text[8] in dict_char_to_int.keys()) and
#             (text[9] in string.digits or text[9] in dict_char_to_int.keys())):
#         return False
#
#     return True
#
#
# def validate_indian_state_code(state_code):
#     """
#     Validate if the state code is a valid Indian state/UT code.
#
#     Args:
#         state_code (str): Two-letter state code
#
#     Returns:
#         bool: True if valid state code, False otherwise
#     """
#     return state_code.upper() in INDIAN_STATE_CODES
#
#
# def format_license(text):
#     """
#     Format the license plate text by converting characters using the mapping dictionaries.
#     Applies specific mapping for each position based on Indian license plate format.
#
#     Args:
#         text (str): License plate text (without spaces).
#
#     Returns:
#         str: Formatted license plate text with proper spacing.
#     """
#     text = text.replace(' ', '').upper()
#     license_plate_ = ''
#
#     # Mapping for each position in Indian format XX##XX####
#     # Positions 0,1,4,5: Should be letters
#     # Positions 2,3,6,7,8,9: Should be numbers
#     mapping = {
#         0: dict_int_to_char,  # State code - letter
#         1: dict_int_to_char,  # State code - letter
#         2: dict_char_to_int,  # District code - number
#         3: dict_char_to_int,  # District code - number
#         4: dict_int_to_char,  # Series code - letter
#         5: dict_int_to_char,  # Series code - letter
#         6: dict_char_to_int,  # Registration - number
#         7: dict_char_to_int,  # Registration - number
#         8: dict_char_to_int,  # Registration - number
#         9: dict_char_to_int  # Registration - number
#     }
#
#     for i in range(10):
#         if text[i] in mapping[i].keys():
#             license_plate_ += mapping[i][text[i]]
#         else:
#             license_plate_ += text[i]
#
#     # Format with spaces: XX## XX ####
#     formatted = f"{license_plate_[:2]}{license_plate_[2:4]} {license_plate_[4:6]} {license_plate_[6:]}"
#     return formatted
#
#
# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image.
#
#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
#
#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score.
#     """
#     detections = reader.readtext(license_plate_crop)
#
#     for detection in detections:
#         bbox, text, score = detection
#
#         # Clean the text
#         text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
#
#         # Try different variations to handle OCR errors
#         text_variations = [text]
#
#         # Add variation with common OCR substitutions
#         text_corrected = text
#         for char, replacement in dict_char_to_int.items():
#             text_corrected = text_corrected.replace(char, replacement)
#         for char, replacement in dict_int_to_char.items():
#             text_corrected = text_corrected.replace(char, replacement)
#         text_variations.append(text_corrected)
#
#         # Check each variation
#         for text_var in text_variations:
#             if license_complies_format(text_var):
#                 formatted_plate = format_license(text_var)
#
#                 # Additional validation: check if state code is valid
#                 state_code = formatted_plate[:2]
#                 if validate_indian_state_code(state_code):
#                     return formatted_plate, score
#                 else:
#                     # If state code is invalid, still return it but with lower confidence
#                     return formatted_plate, score * 0.8
#
#     return None, None
#
#
# def get_car(license_plate, vehicle_track_ids):
#     """
#     Retrieve the vehicle coordinates and ID based on the license plate coordinates.
#
#     Args:
#         license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
#         vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
#
#     Returns:
#         tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
#     """
#     x1, y1, x2, y2, score, class_id = license_plate
#
#     foundIt = False
#     for j in range(len(vehicle_track_ids)):
#         xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
#
#         if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
#             car_indx = j
#             foundIt = True
#             break
#
#     if foundIt:
#         return vehicle_track_ids[car_indx]
#
#     return -1, -1, -1, -1, -1

# Tesseract OCR Version (Works)
import string
import pytesseract
import re
import cv2

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'T': '7', 'B': '8'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '2': 'Z', '7': 'T', '8': 'B'}

INDIAN_STATE_CODES = [
    'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA',
    'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 'RJ', 'SK', 'TN',
    'TS', 'TR', 'UK', 'UP', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'LD', 'PY'
]


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write(
            'frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                if 'car' in results[frame_nmr][car_id] and 'license_plate' in results[frame_nmr][car_id] and \
                        'text' in results[frame_nmr][car_id]['license_plate']:
                    f.write(f"{frame_nmr},{car_id},"
                            f"[{' '.join(map(str, results[frame_nmr][car_id]['car']['bbox']))}],"
                            f"[{' '.join(map(str, results[frame_nmr][car_id]['license_plate']['bbox']))}],"
                            f"{results[frame_nmr][car_id]['license_plate']['bbox_score']},"
                            f"{results[frame_nmr][car_id]['license_plate']['text']},"
                            f"{results[frame_nmr][car_id]['license_plate']['text_score']}\n")


def license_complies_format(text):
    """
    Check if license plate follows Indian format: XX00XX0000
    - Positions 0,1: Letters (State code - e.g., KA, MH, DL)
    - Positions 2,3: Numbers (District code - e.g., 05, 12)
    - Positions 4,5: Letters (Series - can be ANY letters - e.g., AB, CD, XY)
    - Positions 6,7,8,9: Numbers (Unique identifier - e.g., 1234, 5678)
    """
    text = text.replace(' ', '').upper()
    if len(text) != 10:
        return False

    # Check an exact format: XX00XX0000
    return all([
        text[0] in string.ascii_uppercase,  # First letter
        text[1] in string.ascii_uppercase,  # Second letter
        text[2] in string.digits,  # First digit
        text[3] in string.digits,  # Second digit
        text[4] in string.ascii_uppercase,  # Third letter
        text[5] in string.ascii_uppercase,  # Fourth letter
        text[6] in string.digits,  # Third digit
        text[7] in string.digits,  # Fourth digit
        text[8] in string.digits,  # Fifth digit
        text[9] in string.digits  # Sixth digit
    ])


def validate_indian_state_code(code):
    return code.upper() in INDIAN_STATE_CODES


def format_license(text):
    """
    Format license plate text to XX00XX0000 format and apply character corrections
    based on position-specific rules for Indian license plates:
    - XX: State code (letters)
    - 00: District code (digits)
    - XX: Series code (ANY letters - no restrictions)
    - 0000: Unique number (digits)
    """
    text = text.replace(' ', '').upper()

    # Ensure we have exactly 10 characters
    if len(text) != 10:
        return None

    license_plate_ = ''

    # Position-specific character correction mappings
    # Positions 0,1,4,5 should be letters - convert common OCR mistakes from digits to letters
    # Positions 2,3,6,7,8,9 should be digits - convert common OCR mistakes from letters to digits
    for i in range(10):
        char = text[i]

        if i in [0, 1, 4, 5]:  # Should be letters
            if char in dict_int_to_char:
                license_plate_ += dict_int_to_char[char]
            elif char in string.ascii_uppercase:
                license_plate_ += char
            else:
                # Invalid character for letter position
                return None
        else:  # Positions 2,3,6,7,8,9 should be digits
            if char in dict_char_to_int:
                license_plate_ += dict_char_to_int[char]
            elif char in string.digits:
                license_plate_ += char
            else:
                # Invalid character for digit position
                return None

    # Final validation of the corrected format
    if not license_complies_format(license_plate_):
        return None

    return f"{license_plate_[:2]}{license_plate_[2:4]}{license_plate_[4:6]}{license_plate_[6:]}"


def read_license_plate(license_plate_crop):
    try:
        # Check if the image is valid and has content
        if license_plate_crop is None or license_plate_crop.size == 0:
            return None, None

        # Apply additional preprocessing to improve OCR
        # Resize image to improve OCR accuracy
        height, width = license_plate_crop.shape
        if height < 50:  # If the image is too small, resize it
            scale = 50 / height
            new_width = int(width * scale)
            license_plate_crop = cv2.resize(license_plate_crop, (new_width, 50))

        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        license_plate_crop = cv2.morphologyEx(license_plate_crop, cv2.MORPH_CLOSE, kernel)

        # Use Tesseract with multiple configurations
        configs = [
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]

        for config in configs:
            try:
                text = pytesseract.image_to_string(license_plate_crop, config=config)
                text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace('\n', '').strip()

                # Remove any non-alphanumeric characters
                text = re.sub(r'[^A-Z0-9]', '', text)

                # Only process if we have exactly 10 characters (Indian license plate length)
                if len(text) == 10:
                    # Try to format and validate the license plate
                    formatted_license = format_license(text)

                    if formatted_license is not None:
                        # Double-check the format compliance
                        clean_text = formatted_license.replace(' ', '')
                        if license_complies_format(clean_text):
                            # Validate state code (optional - don't reject if state code is invalid due to OCR errors)
                            state_code = clean_text[:2]
                            if validate_indian_state_code(state_code):
                                return formatted_license, 0.9  # High confidence for valid state code
                            else:
                                return formatted_license, 0.8  # Still good confidence even with invalid state code

                # Try with different length variations (in case OCR missed/added characters)
                for length_variation in [9, 11, 8, 12]:
                    if len(text) == length_variation:
                        # Try to pad or trim to 10 characters
                        if len(text) < 10:
                            # Try padding with common characters, but this is risky
                            continue
                        elif len(text) > 10:
                            # Try removing extra characters from start or end
                            for start_trim in range(len(text) - 9):
                                trimmed = text[start_trim:start_trim + 10]
                                formatted_license = format_license(trimmed)
                                if formatted_license is not None:
                                    clean_text = formatted_license.replace(' ', '')
                                    if license_complies_format(clean_text):
                                        return formatted_license, 0.6  # Lower confidence due to trimming

            except Exception as e:
                continue

        return None, None

    except Exception as e:
        print(f"Error in read_license_plate: {e}")
        return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
