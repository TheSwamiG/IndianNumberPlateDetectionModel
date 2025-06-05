from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from pathlib import Path


class LicensePlateDetector:
    def __init__(self, vehicle_model_path='yolo11n.pt', license_plate_model_path='Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt'):
        """
        Initialize the license plate detector

        Args:
            vehicle_model_path: Path to YOLO model for vehicle detection
            license_plate_model_path: Path to custom YOLO model for license plate detection
        """
        self.vehicle_model = YOLO(vehicle_model_path)

        if license_plate_model_path and os.path.exists(license_plate_model_path):
            self.license_plate_model = YOLO(license_plate_model_path)
        else:
            print("Warning: License plate model not found. Please provide a valid path.")
            self.license_plate_model = None

        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def get_car_for_license_plate(self, license_plate_bbox, vehicle_detections):
        """
        Find which vehicle contains the detected license plate

        Args:
            license_plate_bbox: [x1, y1, x2, y2] of license plate
            vehicle_detections: List of vehicle detections with their IDs

        Returns:
            Vehicle bbox and ID, or (-1, -1, -1, -1, -1) if not found
        """
        x1, y1, x2, y2 = license_plate_bbox

        for vehicle_detection in vehicle_detections:
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle_detection

            # Check if license plate is inside the vehicle bbox
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return xcar1, ycar1, xcar2, ycar2, car_id

        return -1, -1, -1, -1, -1

    def detect_vehicles(self, image):
        """
        Detect vehicles in the image

        Args:
            image: Input image (numpy array)

        Returns:
            List of vehicle detections: [[x1, y1, x2, y2, vehicle_id], ...]
        """
        detections = self.vehicle_model(image)[0]
        vehicle_detections = []

        for i, detection in enumerate(detections.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = detection

            # Filter for vehicle classes only
            if int(class_id) in self.vehicle_classes:
                vehicle_detections.append([x1, y1, x2, y2, i])

        return vehicle_detections

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

    def crop_license_plate(self, image, license_plate_bbox, padding=15):
        """
        Crop license plate from image with padding

        Args:
            image: Input image
            license_plate_bbox: [x1, y1, x2, y2] coordinates
            padding: Padding around the license plate

        Returns:
            Cropped license plate image, crop coordinates
        """
        x1, y1, x2, y2 = license_plate_bbox

        # Add padding with boundary checks
        h, w = image.shape[:2]
        y1_crop = max(0, int(y1) - padding)
        y2_crop = min(h, int(y2) + padding)
        x1_crop = max(0, int(x1) - padding)
        x2_crop = min(w, int(x2) + padding)

        # Crop the license plate
        cropped_plate = image[y1_crop:y2_crop, x1_crop:x2_crop]

        # Return None if crop is too small
        if cropped_plate.shape[0] < 20 or cropped_plate.shape[1] < 50:
            return None, None

        return cropped_plate, (x1_crop, y1_crop, x2_crop, y2_crop)

    def process_single_image(self, image_path, output_dir="cropped_plates", save_metadata=True):
        """
        Process a single image to detect and crop license plates

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

        # Detect vehicles and license plates
        vehicle_detections = self.detect_vehicles(image)
        license_plate_detections = self.detect_license_plates(image)

        results = {
            'image_path': image_path,
            'vehicles': vehicle_detections,
            'license_plates': [],
            'cropped_plates': []
        }

        print(f"Found {len(vehicle_detections)} vehicles and {len(license_plate_detections)} license plates")

        # Process each license plate
        for i, lp_detection in enumerate(license_plate_detections):
            x1, y1, x2, y2, score = lp_detection

            # Find associated vehicle
            xcar1, ycar1, xcar2, ycar2, car_id = self.get_car_for_license_plate(
                [x1, y1, x2, y2], vehicle_detections
            )

            if car_id != -1:  # If license plate is associated with a vehicle
                # Crop license plate
                cropped_plate, crop_coords = self.crop_license_plate(
                    image, [x1, y1, x2, y2]
                )

                if cropped_plate is not None:
                    # Save cropped license plate
                    crop_filename = f"{base_name}_car_{car_id}_plate_{i}.jpg"
                    crop_path = os.path.join(output_dir, crop_filename)
                    cv2.imwrite(crop_path, cropped_plate)

                    # Store results
                    plate_info = {
                        'plate_id': i,
                        'car_id': car_id,
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'vehicle_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'detection_score': score,
                        'crop_path': crop_path,
                        'crop_coords': crop_coords
                    }

                    results['license_plates'].append(plate_info)
                    results['cropped_plates'].append(crop_path)

                    print(f"Saved cropped plate: {crop_filename}")

        # Save metadata if requested
        if save_metadata and results['license_plates']:
            metadata_path = os.path.join(output_dir, f"{base_name}_detection_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved metadata: {metadata_path}")

        return results

    def process_multiple_images(self, image_paths, output_dir="cropped_plates", save_metadata=True):
        """
        Process multiple images

        Args:
            image_paths: List of image paths
            output_dir: Directory to save cropped plates
            save_metadata: Whether to save detection metadata

        Returns:
            Dictionary containing all results
        """
        all_results = {}

        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

            result = self.process_single_image(image_path, output_dir, save_metadata=False)
            if result:
                all_results[image_path] = result

        # Save combined metadata
        if save_metadata and all_results:
            metadata_path = os.path.join(output_dir, "batch_detection_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved batch metadata: {metadata_path}")

        return all_results

    def process_folder(self, folder_path, output_dir="cropped_plates",
                       image_extensions=None, save_metadata=True):
        """
        Process all images in a folder

        Args:
            folder_path: Path to folder containing images
            output_dir: Directory to save cropped plates
            image_extensions: List of valid image extensions
            save_metadata: Whether to save detection metadata

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

        return self.process_multiple_images(image_paths, output_dir, save_metadata)


def main():
    """
    Example usage of the LicensePlateDetector
    """
    # Initialize detector (update paths as needed)
    detector = LicensePlateDetector(
        vehicle_model_path='yolo11n.pt',
        license_plate_model_path='/Users/swaminathang/yolov5/runs/detect/train15/weights/best.pt'
    )

    # Example 1: Process single image
    print("=== Processing Single Image ===")
    result = detector.process_single_image('TestRun1.jpeg', 'cropped_plates')

    if result:
        print(f"Successfully processed image. Found {len(result['license_plates'])} license plates.")
        for plate in result['license_plates']:
            print(f"  - Car {plate['car_id']}: {plate['crop_path']}")

    # Example 2: Process multiple images
    # print("\n=== Processing Multiple Images ===")
    # image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # results = detector.process_multiple_images(image_list, 'cropped_plates')

    # Example 3: Process folder
    # print("\n=== Processing Image Folder ===")
    # results = detector.process_folder('/path/to/image/folder', 'cropped_plates')


if __name__ == "__main__":
    main()
