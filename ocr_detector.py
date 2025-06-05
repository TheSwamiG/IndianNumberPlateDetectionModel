import cv2
import numpy as np
import pytesseract
import string
import re
import os
import json
from pathlib import Path


class LicensePlateOCRReader:
    def __init__(self):
        """
        Initialize the OCR reader with character correction dictionaries
        """
        # Character correction mappings
        self.dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'T': '7',
                                 'B': '8'}
        self.dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '2': 'Z', '7': 'T',
                                 '8': 'B'}

        # Indian state codes for validation
        self.indian_state_codes = [
            'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA',
            'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 'RJ', 'SK', 'TN',
            'TS', 'TR', 'UK', 'UP', 'WB', 'AN', 'CH', 'DN', 'DD', 'DL', 'LD', 'PY'
        ]

    def order_points(self, pts):
        """
        Order points in the order: top-left, top-right, bottom-right, bottom-left
        """
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

    def detect_corners_and_apply_perspective_transform(self, license_plate_crop):
        """
        Detect corners of the license plate and apply perspective transformation
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

            # Find the largest rectangular contour
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
            if largest_contour is not None and max_area > 100:
                # Order the points
                points = largest_contour.reshape(4, 2)
                ordered_points = self.order_points(points)

                # Calculate dimensions
                width_top = np.linalg.norm(ordered_points[1] - ordered_points[0])
                width_bottom = np.linalg.norm(ordered_points[2] - ordered_points[3])
                height_left = np.linalg.norm(ordered_points[3] - ordered_points[0])
                height_right = np.linalg.norm(ordered_points[2] - ordered_points[1])

                avg_width = (width_top + width_bottom) / 2
                avg_height = (height_left + height_right) / 2

                # Maintain aspect ratio
                aspect_ratio = avg_width / avg_height if avg_height > 0 else 4.5

                # Set minimum dimensions for better OCR
                min_height = 80
                output_height = max(min_height, int(avg_height))
                output_width = int(output_height * aspect_ratio)

                # Ensure reasonable maximum size
                if output_width > 600:
                    output_width = 600
                    output_height = int(output_width / aspect_ratio)

                # Define destination points
                dst_points = np.array([
                    [0, 0],
                    [output_width, 0],
                    [output_width, output_height],
                    [0, output_height]
                ], dtype=np.float32)

                # Calculate and apply perspective transformation
                matrix = cv2.getPerspectiveTransform(ordered_points.astype(np.float32), dst_points)
                transformed = cv2.warpPerspective(license_plate_crop, matrix, (output_width, output_height))

                return transformed, True

            # If no good contour found, return original
            return license_plate_crop, False

        except Exception as e:
            print(f"Error in perspective transformation: {e}")
            return license_plate_crop, False

    def enhance_license_plate_image(self, license_plate_crop):
        """
        Apply comprehensive image enhancement for better OCR results
        """
        try:
            # Apply perspective transformation first
            transformed_img, transform_applied = self.detect_corners_and_apply_perspective_transform(license_plate_crop)

            # Convert to grayscale
            if len(transformed_img.shape) == 3:
                gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = transformed_img.copy()

            # Resize if too small
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

            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            # Apply unsharp masking
            gaussian = cv2.GaussianBlur(cleaned, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(cleaned, 1.5, gaussian, -0.5, 0)

            return unsharp_mask, transform_applied

        except Exception as e:
            print(f"Error in image enhancement: {e}")
            if len(license_plate_crop.shape) == 3:
                return cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY), False
            return license_plate_crop, False

    def preprocess_for_ocr(self, license_plate_crop):
        """
        Additional preprocessing specifically for OCR optimization
        """
        try:
            # Ensure grayscale
            if len(license_plate_crop.shape) == 3:
                gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = license_plate_crop.copy()

            # Resize for optimal OCR
            height, width = gray.shape
            if height < 64:
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

    def license_complies_format(self, text):
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

    def validate_indian_state_code(self, code):
        """
        Validate if the state code is a valid Indian state code
        """
        return code.upper() in self.indian_state_codes

    def format_license(self, text):
        """
        Format license plate text and apply character corrections
        """
        text = text.replace(' ', '').upper()

        if len(text) != 10:
            return None

        license_plate_ = ''

        for i in range(10):
            char = text[i]

            if i in [0, 1, 4, 5]:  # Should be letters
                if char in self.dict_int_to_char:
                    license_plate_ += self.dict_int_to_char[char]
                elif char in string.ascii_uppercase:
                    license_plate_ += char
                else:
                    return None
            else:  # Should be digits
                if char in self.dict_char_to_int:
                    license_plate_ += self.dict_char_to_int[char]
                elif char in string.digits:
                    license_plate_ += char
                else:
                    return None

        if not self.license_complies_format(license_plate_):
            return None

        return f"{license_plate_[:2]}{license_plate_[2:4]}{license_plate_[4:6]}{license_plate_[6:]}"

    def calculate_confidence_score(self, text, image):
        """
        Calculate confidence score for the OCR result
        """
        try:
            base_score = 0.5

            # Check state code validity
            if len(text) >= 2:
                state_code = text[:2]
                if self.validate_indian_state_code(state_code):
                    base_score += 0.3
                else:
                    base_score += 0.1

            # Check format compliance
            if self.license_complies_format(text):
                base_score += 0.2

            # Image quality factors
            height, width = image.shape

            # Bonus for larger images
            if height >= 50:
                base_score += 0.1

            # Check image contrast
            contrast = image.std()
            if contrast > 30:
                base_score += 0.1
            elif contrast > 15:
                base_score += 0.05

            return min(base_score, 1.0)

        except Exception as e:
            return 0.5

    def read_license_plate(self, license_plate_crop):
        """
        Enhanced OCR function with multiple configurations
        """
        try:
            if license_plate_crop is None or license_plate_crop.size == 0:
                return None, None

            # Apply preprocessing
            preprocessed = self.preprocess_for_ocr(license_plate_crop)

            # Multiple Tesseract configurations
            configs = [
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 2 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]

            best_text = None
            best_score = 0

            for config in configs:
                try:
                    text = pytesseract.image_to_string(preprocessed, config=config)
                    text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace('\n', '').strip()
                    text = re.sub(r'[^A-Z0-9]', '', text)

                    # Process exact 10 character results
                    if len(text) == 10:
                        formatted_license = self.format_license(text)

                        if formatted_license is not None:
                            clean_text = formatted_license.replace(' ', '')
                            if self.license_complies_format(clean_text):
                                confidence = self.calculate_confidence_score(text, preprocessed)

                                if confidence > best_score:
                                    best_text = formatted_license
                                    best_score = confidence

                    # Handle length variations
                    elif 8 <= len(text) <= 12:
                        # Try padding for 9-character text
                        if len(text) == 9:
                            for pos in [2, 3, 6, 7, 8, 9]:
                                for digit in '0123456789':
                                    test_text = text[:pos] + digit + text[pos:]
                                    if len(test_text) == 10:
                                        formatted_license = self.format_license(test_text)
                                        if formatted_license is not None:
                                            clean_text = formatted_license.replace(' ', '')
                                            if self.license_complies_format(clean_text):
                                                confidence = self.calculate_confidence_score(test_text,
                                                                                             preprocessed) * 0.7
                                                if confidence > best_score:
                                                    best_text = formatted_license
                                                    best_score = confidence

                        # Try trimming for longer text
                        elif len(text) > 10:
                            for start_trim in range(len(text) - 9):
                                trimmed = text[start_trim:start_trim + 10]
                                formatted_license = self.format_license(trimmed)
                                if formatted_license is not None:
                                    clean_text = formatted_license.replace(' ', '')
                                    if self.license_complies_format(clean_text):
                                        confidence = self.calculate_confidence_score(trimmed, preprocessed) * 0.8
                                        if confidence > best_score:
                                            best_text = formatted_license
                                            best_score = confidence

                except Exception as e:
                    continue

            return best_text, best_score

        except Exception as e:
            print(f"Error in read_license_plate: {e}")
            return None, None

    def process_cropped_plate(self, image_path, apply_perspective_transform=True, save_processed=False):
        """
        Process a single cropped license plate image

        Args:
            image_path: Path to cropped license plate image
            apply_perspective_transform: Whether to apply perspective transformation
            save_processed: Whether to save the processed image

        Returns:
            Dictionary with OCR results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return None

            print(f"Processing: {os.path.basename(image_path)}")

            # Apply enhancement with perspective transformation
            if apply_perspective_transform:
                enhanced_image, transform_applied = self.enhance_license_plate_image(image)
            else:
                # Just apply basic preprocessing without perspective transform
                enhanced_image = self.preprocess_for_ocr(image)
                transform_applied = False

            # Save processed image if requested
            if save_processed:
                processed_filename = f"processed_{os.path.basename(image_path)}"
                cv2.imwrite(processed_filename, enhanced_image)
                print(f"Saved processed image: {processed_filename}")

            # Try multiple threshold values for better OCR
            threshold_values = [0, 64, 128, 180, 220]
            best_text = None
            best_score = 0

            # Try the enhanced image without additional thresholding
            license_plate_text, license_plate_text_score = self.read_license_plate(enhanced_image)
            if license_plate_text is not None and license_plate_text_score > best_score:
                best_text = license_plate_text
                best_score = license_plate_text_score

            # Try with different threshold values
            for thresh_val in threshold_values:
                if thresh_val == 0:
                    # Adaptive threshold
                    license_plate_thresh = cv2.adaptiveThreshold(
                        enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                else:
                    # Fixed threshold
                    _, license_plate_thresh = cv2.threshold(
                        enhanced_image, thresh_val, 255, cv2.THRESH_BINARY
                    )

                # Also try inverted threshold
                license_plate_thresh_inv = cv2.bitwise_not(license_plate_thresh)

                # Test both regular and inverted threshold
                for thresh_img in [license_plate_thresh, license_plate_thresh_inv]:
                    license_plate_text, license_plate_text_score = self.read_license_plate(thresh_img)
                    if license_plate_text is not None and license_plate_text_score > best_score:
                        best_text = license_plate_text
                        best_score = license_plate_text_score

            # Prepare result
            result = {
                'image_path': image_path,
                'license_plate_text': best_text,
                'confidence_score': best_score,
                'perspective_transform_applied': transform_applied,
                'success': best_text is not None
            }

            if best_text:
                transform_msg = "with perspective correction" if transform_applied else "without perspective correction"
                print(f"✓ License plate detected: '{best_text}' with confidence {best_score:.3f} {transform_msg}")
            else:
                print(f"✗ No valid license plate detected in {os.path.basename(image_path)}")

            return result

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'license_plate_text': None,
                'confidence_score': 0.0,
                'perspective_transform_applied': False,
                'success': False,
                'error': str(e)
            }

    def process_multiple_cropped_plates(self, image_paths, apply_perspective_transform=True, save_processed=False):
        """
        Process multiple cropped license plate images

        Args:
            image_paths: List of paths to cropped license plate images
            apply_perspective_transform: Whether to apply perspective transformation
            save_processed: Whether to save processed images

        Returns:
            List of dictionaries with OCR results
        """
        results = []

        print(f"Processing {len(image_paths)} license plate images...")

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")

            result = self.process_cropped_plate(
                image_path,
                apply_perspective_transform=apply_perspective_transform,
                save_processed=save_processed
            )

            results.append(result)

        # Print summary
        successful_reads = sum(1 for r in results if r['success'])
        print(f"\n=== Processing Summary ===")
        print(f"Total images processed: {len(image_paths)}")
        print(f"Successful OCR reads: {successful_reads}")
        print(f"Success rate: {successful_reads / len(image_paths) * 100:.1f}%")

        return results

    def process_folder_of_cropped_plates(self, folder_path, apply_perspective_transform=True, save_processed=False,
                                         image_extensions=None):
        """
        Process all cropped license plate images in a folder

        Args:
            folder_path: Path to folder containing cropped license plate images
            apply_perspective_transform: Whether to apply perspective transformation
            save_processed: Whether to save processed images
            image_extensions: List of image file extensions to process

        Returns:
            List of dictionaries with OCR results
        """
        try:
            if image_extensions is None:
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

            # Validate folder path
            if not os.path.exists(folder_path):
                print(f"Error: Folder {folder_path} does not exist")
                return []

            if not os.path.isdir(folder_path):
                print(f"Error: {folder_path} is not a directory")
                return []

            # Get all image files from folder
            image_paths = []
            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(folder_path, filename))

            if not image_paths:
                print(f"No image files found in {folder_path}")
                print(f"Supported extensions: {image_extensions}")
                return []

            print(f"Found {len(image_paths)} images in {folder_path}")
            image_paths.sort()  # Sort for consistent processing order

            return self.process_multiple_cropped_plates(
                image_paths,
                apply_perspective_transform=apply_perspective_transform,
                save_processed=save_processed
            )

        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            return []

    def save_results_to_json(self, results, output_path):
        """
        Save OCR results to JSON file

        Args:
            results: List of OCR result dictionaries
            output_path: Path to save JSON file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

    def save_results_to_csv(self, results, output_path):
        """
        Save OCR results to CSV file

        Args:
            results: List of OCR result dictionaries
            output_path: Path to save CSV file
        """
        try:
            import csv

            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow([
                    'image_path', 'license_plate_text', 'confidence_score',
                    'perspective_transform_applied', 'success'
                ])

                # Write data
                for result in results:
                    writer.writerow([
                        result.get('image_path', ''),
                        result.get('license_plate_text', ''),
                        result.get('confidence_score', 0.0),
                        result.get('perspective_transform_applied', False),
                        result.get('success', False)
                    ])

            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the OCR reader
    ocr_reader = LicensePlateOCRReader()

    # # Example 1: Process a single cropped license plate image
    # print("=== Example 1: Single Image Processing ===")
    # single_result = ocr_reader.process_cropped_plate(
    #     'cropped_license_plate.jpg',
    #     apply_perspective_transform=True,
    #     save_processed=True
    # )
    # if single_result:
    #     print(f"Result: {single_result}")
    #
    # # Example 2: Process multiple cropped license plate images
    # print("\n=== Example 2: Multiple Images Processing ===")
    # image_list = [
    #     'plate1.jpg',
    #     'plate2.jpg',
    #     'plate3.jpg'
    # ]
    # multiple_results = ocr_reader.process_multiple_cropped_plates(
    #     image_list,
    #     apply_perspective_transform=True,
    #     save_processed=False
    # )
    #
    # # Save results
    # if multiple_results:
    #     ocr_reader.save_results_to_json(multiple_results, 'ocr_results.json')
    #     ocr_reader.save_results_to_csv(multiple_results, 'ocr_results.csv')

    # Example 3: Process all images in a folder
    print("\n=== Example 3: Folder Processing ===")
    folder_results = ocr_reader.process_folder_of_cropped_plates(
        'cropped_plates',
        apply_perspective_transform=True,
        save_processed=False
    )

    if folder_results:
        ocr_reader.save_results_to_json(folder_results, 'folder_ocr_results.json')
        ocr_reader.save_results_to_csv(folder_results, 'folder_ocr_results.csv')

    print("\nOCR Reader module loaded successfully!")
    print("Use LicensePlateOCRReader class methods to process your cropped license plate images.")
