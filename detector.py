from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from pathlib import Path
import concurrent.futures
from threading import Lock
import time


class LicensePlateDetector:
    def __init__(self, license_plate_model_path='Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt'):
        """
        Initialize the license plate detector

        Args:
            license_plate_model_path: Path to custom YOLO model for license plate detection
        """
        if license_plate_model_path and os.path.exists(license_plate_model_path):
            self.license_plate_model = YOLO(license_plate_model_path)
        else:
            print("Warning: License plate model not found. Please provide a valid path.")
            self.license_plate_model = None

        # Thread lock for thread-safe operations
        self.lock = Lock()

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
        Always applies transformation - if no clear corners found, uses image boundaries
        """
        try:
            # Convert to grayscale for corner detection
            gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            h, w = license_plate_crop.shape[:2]

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest rectangular contour
            largest_contour = None
            max_area = 0
            corner_detection_used = False

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

            # If we found a good rectangular contour, use it
            if largest_contour is not None and max_area > 100:
                # Order the points
                points = largest_contour.reshape(4, 2)
                ordered_points = self.order_points(points)
                corner_detection_used = True
            else:
                # If no good contour found, use image boundaries as fallback
                # This ensures we ALWAYS apply perspective transformation
                ordered_points = np.array([
                    [0, 0],  # top-left
                    [w - 1, 0],  # top-right
                    [w - 1, h - 1],  # bottom-right
                    [0, h - 1]  # bottom-left
                ], dtype=np.float32)
                corner_detection_used = False

            # Calculate dimensions
            width_top = np.linalg.norm(ordered_points[1] - ordered_points[0])
            width_bottom = np.linalg.norm(ordered_points[2] - ordered_points[3])
            height_left = np.linalg.norm(ordered_points[3] - ordered_points[0])
            height_right = np.linalg.norm(ordered_points[2] - ordered_points[1])

            avg_width = (width_top + width_bottom) / 2
            avg_height = (height_left + height_right) / 2

            # Maintain an aspect ratio
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

            # Always return transformed image with status of corner detection
            return transformed, corner_detection_used

        except Exception as e:
            print(f"Error in perspective transformation: {e}")
            # Even in case of error, try to return a resized version
            try:
                h, w = license_plate_crop.shape[:2]
                aspect_ratio = w / h if h > 0 else 4.5
                min_height = 80
                output_height = max(min_height, h)
                output_width = int(output_height * aspect_ratio)

                if output_width > 600:
                    output_width = 600
                    output_height = int(output_width / aspect_ratio)

                resized = cv2.resize(license_plate_crop, (output_width, output_height))
                return resized, False
            except:
                return license_plate_crop, False

    def detect_license_plates(self, image, min_confidence=0.25):
        """
        Detect license plates in the image

        Args:
            image: Input image (numpy array)
            min_confidence: Minimum confidence threshold for detection

        Returns:
            List of license plate detections: [[x1, y1, x2, y2, score], ...]
        """
        if self.license_plate_model is None:
            print("Error: License plate model not loaded")
            return []

        detections = self.license_plate_model(image)[0]
        license_plate_detections = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Filter by confidence
            if score >= min_confidence:
                license_plate_detections.append([x1, y1, x2, y2, score])

        return license_plate_detections

    def crop_and_transform_license_plate(self, image, license_plate_bbox):
        """
        Crop license plate from image and apply perspective transformation
        Always applies perspective transformation

        Args:
            image: Input image
            license_plate_bbox: [x1, y1, x2, y2] coordinates

        Returns:
            Cropped and transformed license plate image, crop coordinates, transformation details
        """
        x1, y1, x2, y2 = license_plate_bbox

        # Clip coordinates to image boundaries
        h, w = image.shape[:2]
        y1_crop = max(0, int(y1))
        y2_crop = min(h, int(y2))
        x1_crop = max(0, int(x1))
        x2_crop = min(w, int(x2))

        # Crop the license plate
        cropped_plate = image[y1_crop:y2_crop, x1_crop:x2_crop]

        # Return None if crop is too small
        if cropped_plate.shape[0] < 20 or cropped_plate.shape[1] < 50:
            return None, None, {'applied': False, 'corner_detection_used': False}

        # Always apply perspective transformation
        transformed_plate, corner_detection_used = self.detect_corners_and_apply_perspective_transform(cropped_plate)

        # Return transformation details
        transform_details = {
            'applied': True,  # Always True now
            'corner_detection_used': corner_detection_used
        }

        return transformed_plate, (x1_crop, y1_crop, x2_crop, y2_crop), transform_details

    def process_single_image(self, image_path, output_dir="cropped_plates", save_metadata=True):
        """
        Process a single image to detect and crop license plates with perspective transformation

        Args:
            image_path: Path to input image
            output_dir: Directory to save cropped plates
            save_metadata: Whether to save detection metadata

        Returns:
            Dictionary containing detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get base filename without extension
        base_name = Path(image_path).stem

        # Detect license plates
        license_plate_detections = self.detect_license_plates(image)

        results = {
            'image_path': image_path,
            'image_shape': image.shape,
            'license_plates': [],
            'cropped_plates': []
        }

        print(f"Found {len(license_plate_detections)} license plates")

        # Process each license plate
        for i, lp_detection in enumerate(license_plate_detections):
            x1, y1, x2, y2, score = lp_detection

            # Crop and transform license plate
            transformed_plate, crop_coords, transform_details = self.crop_and_transform_license_plate(
                image, [x1, y1, x2, y2]
            )

            if transformed_plate is not None:
                # Save cropped and transformed license plate
                crop_filename = f"{base_name}_plate_{i}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, transformed_plate)

                # Store results with enhanced transformation details
                plate_info = {
                    'plate_id': i,
                    'license_plate_bbox': [x1, y1, x2, y2],
                    'detection_score': score,
                    'crop_path': crop_path,
                    'crop_coords': crop_coords,
                    'perspective_transform_applied': transform_details['applied'],
                    'corner_detection_used': transform_details['corner_detection_used']
                }

                results['license_plates'].append(plate_info)
                results['cropped_plates'].append(crop_path)

                # Enhanced status message
                if transform_details['corner_detection_used']:
                    transform_status = "with corner-based perspective correction"
                else:
                    transform_status = "with boundary-based perspective correction"

                print(f"Saved cropped plate: {crop_filename} ({transform_status})")

        # Save metadata if requested
        if save_metadata and results['license_plates']:
            metadata_path = os.path.join(output_dir, f"{base_name}_detection_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved metadata: {metadata_path}")

        return results

    def process_single_image_worker(self, args):
        """
        Worker function for multi-threading
        """
        image_path, output_dir, save_metadata = args
        try:
            return self.process_single_image(image_path, output_dir, save_metadata=False)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_multiple_images(self, image_paths, output_dir="cropped_plates", save_metadata=True, max_workers=4):
        """
        Process multiple images using multi-threading

        Args:
            image_paths: List of image paths
            output_dir: Directory to save cropped plates
            save_metadata: Whether to save detection metadata
            max_workers: Maximum number of threads to use

        Returns:
            Dictionary containing all results
        """
        all_results = {}

        print(f"Processing {len(image_paths)} images using {max_workers} threads...")
        start_time = time.time()

        # Prepare arguments for workers
        args_list = [(image_path, output_dir, False) for image_path in image_paths]

        # Use ThreadPoolExecutor for multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(self.process_single_image_worker, args): args[0]
                              for args in args_list}

            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        with self.lock:
                            all_results[image_path] = result

                    # Progress update
                    progress = (i + 1) / len(image_paths) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(image_paths)}) - {os.path.basename(image_path)}")

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")

        # Save combined metadata
        if save_metadata and all_results:
            metadata_path = os.path.join(output_dir, "batch_detection_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved batch metadata: {metadata_path}")

        return all_results

    def process_folder(self, folder_path, output_dir="cropped_plates",
                       image_extensions=None, save_metadata=True, max_workers=4):
        """
        Process all images in a folder using multi-threading

        Args:
            folder_path: Path to folder containing images
            output_dir: Directory to save cropped plates
            image_extensions: List of valid image extensions
            save_metadata: Whether to save detection metadata
            max_workers: Maximum number of threads to use

        Returns:
            Dictionary containing all results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

        # Validate folder
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Error: Invalid folder path {folder_path}")
            return None

        # Get all image files
        image_paths = []
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(folder_path, filename))

        if not image_paths:
            print(f"No image files found in {folder_path}")
            return None

        print(f"Found {len(image_paths)} images to process")
        image_paths.sort()  # Sort for consistent processing

        return self.process_multiple_images(image_paths, output_dir, save_metadata, max_workers)

    def detect_and_get_results(self, image_path):
        """
        Get detection results without saving cropped images
        Used for visualization purposes

        Args:
            image_path: Path to input image

        Returns:
            Dictionary containing detection results with original image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # Detect license plates
        license_plate_detections = self.detect_license_plates(image)

        results = {
            'original_image': image,
            'image_path': image_path,
            'image_shape': image.shape,
            'license_plates': [],
            'license_plate_detections': license_plate_detections  # Raw detections
        }

        # Process each license plate
        for i, lp_detection in enumerate(license_plate_detections):
            x1, y1, x2, y2, score = lp_detection

            # Crop and transform license plate for OCR processing
            transformed_plate, crop_coords, transform_details = self.crop_and_transform_license_plate(
                image, [x1, y1, x2, y2]
            )

            if transformed_plate is not None:
                # Store results with enhanced transformation details
                plate_info = {
                    'plate_id': i,
                    'license_plate_bbox': [x1, y1, x2, y2],
                    'detection_score': score,
                    'cropped_plate': transformed_plate,  # Store transformed image for OCR
                    'crop_coords': crop_coords,
                    'perspective_transform_applied': transform_details['applied'],
                    'corner_detection_used': transform_details['corner_detection_used']
                }

                results['license_plates'].append(plate_info)

        return results


def main():
    """
    Example usage of the enhanced LicensePlateDetector
    """
    # Initialize detector (update paths as needed)
    detector = LicensePlateDetector(
        license_plate_model_path='/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt'
    )

    # Example 1: Process single image
    print("=== Processing Single Image ===")
    result = detector.process_single_image(
        '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/NumberPlateImages/TestRun2.jpeg',
        'cropped_plates'
    )

    if result:
        print(f"Successfully processed image. Found {len(result['license_plates'])} license plates.")
        for plate in result['license_plates']:
            print(f"  - Plate {plate['plate_id']}: {plate['crop_path']}")
            print(f"    Corner detection: {plate['corner_detection_used']}")
            print(f"    Perspective transform: {plate['perspective_transform_applied']}")

    # Example 2: Process multiple images with multi-threading
    print("\n=== Processing Multiple Images (Multi-threaded) ===")
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # results = detector.process_multiple_images(image_list, 'cropped_plates', max_workers=4)

    # Example 3: Process folder with multi-threading
    print("\n=== Processing Image Folder (Multi-threaded) ===")
    # results = detector.process_folder('/path/to/your/images', 'cropped_plates', max_workers=4)

    # Example 4: Process for visualization
    print("\n=== Get Results for Visualization ===")
    viz_results = detector.detect_and_get_results(
        '/Users/swaminathang/PycharmProjects/IndianNumberplateDetection/NumberPlateImages/frame-190.jpg'
    )

    if viz_results:
        print(f"Found {len(viz_results['license_plates'])} license plates for visualization")
        for plate in viz_results['license_plates']:
            print(f"  - Plate {plate['plate_id']}: Score {plate['detection_score']:.2f}")
            print(f"    Corner detection: {plate['corner_detection_used']}")
            print(f"    Perspective transform: {plate['perspective_transform_applied']}")


if __name__ == "__main__":
    main()
