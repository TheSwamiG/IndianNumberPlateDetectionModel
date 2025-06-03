# import string
# import pytesseract
# import re
# import cv2
# import os
#
# # Common OCR misclassifications (letter ↔ number)
# dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'T': '7', 'B': '8'}
# dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '2': 'Z', '7': 'T', '8': 'B'}
#
# INDIAN_STATE_CODES = [
#     'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA',
#     'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 'RJ', 'SK', 'TN',
#     'TS', 'TR', 'UK', 'UP', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'LD', 'PY'
# ]
#
#
# def write_csv(results, output_path):
#     with open(output_path, 'w') as f:
#         f.write(
#             'frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
#         for frame_nmr in results:
#             for car_id in results[frame_nmr]:
#                 if 'car' in results[frame_nmr][car_id] and 'license_plate' in results[frame_nmr][car_id] and \
#                         'text' in results[frame_nmr][car_id]['license_plate']:
#                     f.write(f"{frame_nmr},{car_id},"
#                             f"[{' '.join(map(str, results[frame_nmr][car_id]['car']['bbox']))}],"
#                             f"[{' '.join(map(str, results[frame_nmr][car_id]['license_plate']['bbox']))}],"
#                             f"{results[frame_nmr][car_id]['license_plate']['bbox_score']},"
#                             f"{results[frame_nmr][car_id]['license_plate']['text']},"
#                             f"{results[frame_nmr][car_id]['license_plate']['text_score']}\n")
#
#
# def license_complies_format(text):
#     text = text.replace(' ', '').upper()
#     if len(text) != 10:
#         return False
#     return all([
#         text[0] in string.ascii_uppercase,
#         text[1] in string.ascii_uppercase,
#         text[2] in string.digits,
#         text[3] in string.digits,
#         text[4] in string.ascii_uppercase,
#         text[5] in string.ascii_uppercase,
#         text[6] in string.digits,
#         text[7] in string.digits,
#         text[8] in string.digits,
#         text[9] in string.digits
#     ])
#
#
# def validate_indian_state_code(code):
#     return code.upper() in INDIAN_STATE_CODES
#
#
# def format_license(text):
#     text = text.replace(' ', '').upper()
#     if len(text) != 10:
#         return None
#
#     formatted = ''
#     for i in range(10):
#         char = text[i]
#         if i in [0, 1, 4, 5]:  # should be letters
#             formatted += dict_int_to_char.get(char, char) if char.isdigit() else char
#             if not formatted[-1].isalpha():
#                 return None
#         else:  # should be digits
#             formatted += dict_char_to_int.get(char, char) if char.isalpha() else char
#             if not formatted[-1].isdigit():
#                 return None
#
#     if not license_complies_format(formatted):
#         return None
#
#     return formatted
#
#
# def read_license_plate(license_plate_crop):
#     try:
#         if license_plate_crop is None or license_plate_crop.size == 0:
#             return None, None
#
#         height, width = license_plate_crop.shape
#         if height < 50:
#             scale = 50 / height
#             license_plate_crop = cv2.resize(license_plate_crop, (int(width * scale), 50))
#
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         license_plate_crop = cv2.morphologyEx(license_plate_crop, cv2.MORPH_CLOSE, kernel)
#
#         configs = [
#             '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
#         ]
#
#         best_text = None
#         best_score = 0
#
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(license_plate_crop, config=config)
#                 text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace('\n', '').strip()
#                 text = re.sub(r'[^A-Z0-9]', '', text)
#
#                 print(f"OCR raw text: {text}")
#
#                 formatted = format_license(text)
#                 if formatted:
#                     print(f"Formatted license: {formatted}")
#                     return formatted, 0.9
#
#                 # fallback if formatting fails
#                 if len(text) >= 6 and len(text) > len(best_text or ''):
#                     best_text = text
#                     best_score = 0.5
#
#             except Exception as e:
#                 print(f"OCR error: {e}")
#                 continue
#
#         return best_text, best_score if best_text else (None, None)
#
#     except Exception as e:
#         print(f"read_license_plate fatal error: {e}")
#         return None, None
#
#
# def get_car(license_plate, vehicle_track_ids):
#     x1, y1, x2, y2, score, class_id = license_plate
#
#     for j in range(len(vehicle_track_ids)):
#         xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
#
#         if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
#             return vehicle_track_ids[j]
#
#     return -1, -1, -1, -1, -1



# import string
# import pytesseract
# import re
# import cv2
# import os
#
# dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'T': '7', 'B': '8'}
# dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '2': 'Z', '7': 'T', '8': 'B'}
#
# INDIAN_STATE_CODES = [
#     'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA',
#     'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 'RJ', 'SK', 'TN',
#     'TS', 'TR', 'UK', 'UP', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'LD', 'PY'
# ]
#
#
# def write_csv(results, output_path):
#     with open(output_path, 'w') as f:
#         f.write(
#             'frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
#         for frame_nmr in results:
#             for car_id in results[frame_nmr]:
#                 if 'car' in results[frame_nmr][car_id] and 'license_plate' in results[frame_nmr][car_id]:
#                     plate_data = results[frame_nmr][car_id]['license_plate']
#                     text = plate_data.get('text', '')
#                     score = plate_data.get('text_score', 0.0)
#
#                     f.write(f"{frame_nmr},{car_id},"
#                             f"[{' '.join(map(str, results[frame_nmr][car_id]['car']['bbox']))}],"
#                             f"[{' '.join(map(str, plate_data['bbox']))}],"
#                             f"{plate_data['bbox_score']},{text},{score}\n")
#
#
# def license_complies_format(text):
#     text = text.replace(' ', '').upper()
#     if len(text) != 10:
#         return False
#     return all([
#         text[0] in string.ascii_uppercase,
#         text[1] in string.ascii_uppercase,
#         text[2] in string.digits,
#         text[3] in string.digits,
#         text[4] in string.ascii_uppercase,
#         text[5] in string.ascii_uppercase,
#         text[6] in string.digits,
#         text[7] in string.digits,
#         text[8] in string.digits,
#         text[9] in string.digits
#     ])
#
#
# def validate_indian_state_code(code):
#     return code.upper() in INDIAN_STATE_CODES
#
#
# def format_license(text):
#     text = text.replace(' ', '').upper()
#     if len(text) != 10:
#         return None
#
#     license_plate_ = ''
#     for i in range(10):
#         char = text[i]
#         if i in [0, 1, 4, 5]:  # Letters
#             if char in dict_int_to_char:
#                 license_plate_ += dict_int_to_char[char]
#             elif char in string.ascii_uppercase:
#                 license_plate_ += char
#             else:
#                 return None
#         else:  # Digits
#             if char in dict_char_to_int:
#                 license_plate_ += dict_char_to_int[char]
#             elif char in string.digits:
#                 license_plate_ += char
#             else:
#                 return None
#
#     if not license_complies_format(license_plate_):
#         return None
#
#     return license_plate_
#
#
# def read_license_plate(license_plate_crop):
#     try:
#         if license_plate_crop is None or license_plate_crop.size == 0:
#             return None, None
#
#         # Save image for debug
#         debug_dir = "ocr_debug"
#         os.makedirs(debug_dir, exist_ok=True)
#         debug_path = os.path.join(debug_dir, f"crop_{str(hash(license_plate_crop.tobytes()))[:8]}.png")
#         cv2.imwrite(debug_path, license_plate_crop)
#
#         # Preprocessing
#         gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) if len(license_plate_crop.shape) == 3 else license_plate_crop
#         h, w = gray.shape
#         if h < 50:
#             scale = 50 / h
#             new_w = int(w * scale)
#             gray = cv2.resize(gray, (new_w, 50))
#
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
#
#         configs = [
#             '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
#             '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
#         ]
#
#         best_text = None
#         best_score = 0
#
#         for config in configs:
#             try:
#                 text = pytesseract.image_to_string(gray, config=config)
#                 text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace('\n', '').strip()
#                 text = re.sub(r'[^A-Z0-9]', '', text)
#
#                 print(f"OCR raw text: {text}")
#
#                 if len(text) >= 6:
#                     formatted_license = format_license(text)
#                     if formatted_license is not None:
#                         clean_text = formatted_license.replace(' ', '')
#                         if license_complies_format(clean_text):
#                             state_code = clean_text[:2]
#                             if validate_indian_state_code(state_code):
#                                 return formatted_license, 0.9
#                             else:
#                                 return formatted_license, 0.8
#
#                     # Even if not valid format, keep best possible text
#                     if len(text) > len(best_text or ''):
#                         best_text = text
#                         best_score = 0.5  # fallback score
#             except Exception:
#                 continue
#
#         if best_text:
#             print(f"OCR fallback: {best_text}")
#             return best_text, best_score
#
#         return None, None
#
#     except Exception as e:
#         print(f"Error in read_license_plate: {e}")
#         return None, None
#
#
# def get_car(license_plate, vehicle_track_ids):
#     x1, y1, x2, y2, score, class_id = license_plate
#     for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
#         if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
#             return vehicle_track_ids[j]
#     return -1, -1, -1, -1, -1


# Enhanced Utility Functions with Improved OCR for License Plate Detection
import string
import pytesseract
import re
import cv2
import numpy as np

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
            'frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score,perspective_transform\n')
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                if 'car' in results[frame_nmr][car_id] and 'license_plate' in results[frame_nmr][car_id] and \
                        'text' in results[frame_nmr][car_id]['license_plate']:
                    perspective_applied = results[frame_nmr][car_id]['license_plate'].get(
                        'perspective_transform_applied', False)
                    f.write(f"{frame_nmr},{car_id},"
                            f"[{' '.join(map(str, results[frame_nmr][car_id]['car']['bbox']))}],"
                            f"[{' '.join(map(str, results[frame_nmr][car_id]['license_plate']['bbox']))}],"
                            f"{results[frame_nmr][car_id]['license_plate']['bbox_score']},"
                            f"{results[frame_nmr][car_id]['license_plate']['text']},"
                            f"{results[frame_nmr][car_id]['license_plate']['text_score']},"
                            f"{perspective_applied}\n")


def license_complies_format(text):
    """
    Check if license plate follows Indian format: XX00XX0000
    """
    text = text.replace(' ', '').upper()
    if len(text) != 10:
        return False

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
    """
    text = text.replace(' ', '').upper()

    if len(text) != 10:
        return None

    license_plate_ = ''

    for i in range(10):
        char = text[i]

        if i in [0, 1, 4, 5]:  # Should be letters
            if char in dict_int_to_char:
                license_plate_ += dict_int_to_char[char]
            elif char in string.ascii_uppercase:
                license_plate_ += char
            else:
                return None
        else:  # Positions 2,3,6,7,8,9 should be digits
            if char in dict_char_to_int:
                license_plate_ += dict_char_to_int[char]
            elif char in string.digits:
                license_plate_ += char
            else:
                return None

    if not license_complies_format(license_plate_):
        return None

    return f"{license_plate_[:2]}{license_plate_[2:4]}{license_plate_[4:6]}{license_plate_[6:]}"


def preprocess_for_ocr(license_plate_crop):
    """
    Additional preprocessing specifically for OCR optimization
    """
    try:
        # Ensure we have a grayscale image
        if len(license_plate_crop.shape) == 3:
            gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = license_plate_crop.copy()

        # Resize for optimal OCR (Tesseract works best with images at least 300 DPI equivalent)
        height, width = gray.shape
        if height < 64:  # Minimum height for good OCR
            scale_factor = 64 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 64), interpolation=cv2.INTER_CUBIC)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply median blur to reduce salt and pepper noise
        denoised = cv2.medianBlur(filtered, 3)

        return denoised

    except Exception as e:
        print(f"Error in OCR preprocessing: {e}")
        return license_plate_crop


# def preprocess_for_ocr(license_plate_crop):
#     """
#     Additional preprocessing specifically for OCR optimization
#     """
#     try:
#         # Ensure grayscale image
#         if len(license_plate_crop.shape) == 3:
#             gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = license_plate_crop.copy()
#
#         # Resize for optimal OCR (minimum height 64 px)
#         height, width = gray.shape
#         if height < 64:
#             scale_factor = 64 / height
#             new_width = int(width * scale_factor)
#             gray = cv2.resize(gray, (new_width, 64), interpolation=cv2.INTER_CUBIC)
#
#         # Apply bilateral filter to reduce noise while preserving edges
#         filtered = cv2.bilateralFilter(gray, 11, 17, 17)
#
#         # Apply median blur to reduce salt and pepper noise
#         denoised = cv2.medianBlur(filtered, 3)
#
#         # **Adaptive thresholding** - handle uneven lighting
#         adaptive_thresh = cv2.adaptiveThreshold(
#             denoised,
#             255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             11,  # block size (odd number)
#             2    # constant subtracted from mean
#         )
#
#         return adaptive_thresh
#
#     except Exception as e:
#         print(f"Error in OCR preprocessing: {e}")
#         return license_plate_crop
#

def read_license_plate(license_plate_crop):
    """
    Enhanced OCR function with multiple configurations and preprocessing
    """
    try:
        # Check if the image is valid and has content
        if license_plate_crop is None or license_plate_crop.size == 0:
            return None, None

        # Apply additional preprocessing
        preprocessed = preprocess_for_ocr(license_plate_crop)

        # Multiple Tesseract configurations optimized for license plates
        configs = [
            # PSM 8: Single word - best for license plates
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # PSM 7: Single text line
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # PSM 6: Uniform block of text
            '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # PSM 13: Raw line (no heuristics)
            '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # Additional configuration with OCR Engine Mode 3 (Default LSTM)
            '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            # Configuration with both old and new OCR engine
            '--oem 2 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]

        best_text = None
        best_score = 0

        for config in configs:
            try:
                # Try OCR with current configuration
                text = pytesseract.image_to_string(preprocessed, config=config)
                text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace('\n', '').strip()

                # Remove any non-alphanumeric characters
                text = re.sub(r'[^A-Z0-9]', '', text)

                # Only process if we have exactly 10 characters
                if len(text) == 10:
                    formatted_license = format_license(text)

                    if formatted_license is not None:
                        clean_text = formatted_license.replace(' ', '')
                        if license_complies_format(clean_text):
                            # Calculate confidence score based on various factors
                            confidence = calculate_confidence_score(text, preprocessed)

                            if confidence > best_score:
                                best_text = formatted_license
                                best_score = confidence

                # Try with length variations (handle OCR errors)
                elif 8 <= len(text) <= 12:
                    # For shorter text, try padding scenarios (risky but sometimes necessary)
                    if len(text) == 9:
                        # Try adding a digit at common positions
                        for pos in [2, 3, 6, 7, 8, 9]:  # Digit positions
                            for digit in '0123456789':
                                test_text = text[:pos] + digit + text[pos:]
                                if len(test_text) == 10:
                                    formatted_license = format_license(test_text)
                                    if formatted_license is not None:
                                        clean_text = formatted_license.replace(' ', '')
                                        if license_complies_format(clean_text):
                                            confidence = calculate_confidence_score(test_text,
                                                                                    preprocessed) * 0.7  # Lower confidence for padded
                                            if confidence > best_score:
                                                best_text = formatted_license
                                                best_score = confidence

                    # For longer text, try trimming
                    elif len(text) > 10:
                        for start_trim in range(len(text) - 9):
                            trimmed = text[start_trim:start_trim + 10]
                            formatted_license = format_license(trimmed)
                            if formatted_license is not None:
                                clean_text = formatted_license.replace(' ', '')
                                if license_complies_format(clean_text):
                                    confidence = calculate_confidence_score(trimmed,
                                                                            preprocessed) * 0.8  # Lower confidence for trimmed
                                    if confidence > best_score:
                                        best_text = formatted_license
                                        best_score = confidence

            except Exception as e:
                continue

        return best_text, best_score

    except Exception as e:
        print(f"Error in read_license_plate: {e}")
        return None, None


def calculate_confidence_score(text, image):
    """
    Calculate a confidence score for the OCR result based on various factors
    """
    try:
        base_score = 0.5  # Base confidence

        # Check state code validity
        if len(text) >= 2:
            state_code = text[:2]
            if validate_indian_state_code(state_code):
                base_score += 0.3
            else:
                base_score += 0.1  # Some bonus for having letters in right position

        # Check format compliance
        if license_complies_format(text):
            base_score += 0.2

        # Image quality factors
        height, width = image.shape

        # Bonus for larger images (usually better quality)
        if height >= 50:
            base_score += 0.1

        # Check image contrast (higher contrast usually means better OCR)
        contrast = image.std()
        if contrast > 30:  # Good contrast
            base_score += 0.1
        elif contrast > 15:  # Moderate contrast
            base_score += 0.05

        # Ensure score doesn't exceed 1.0
        return min(base_score, 1.0)

    except Exception as e:
        return 0.5  # Default confidence


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
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