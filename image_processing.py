import cv2
import numpy as np
import os
from database_integration import AirQualityDatabase


class ImageProcessor:
    
    def __init__(self, db_path="db_air_quality"):
        self.db = AirQualityDatabase(db_path)
        self.image = None
        self.original_image = None
        self.image_path = None
        self.processing_history = []
    
    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.image_path = image_path
        self.processing_history = []
        
        height, width = self.image.shape[:2]
        channels = self.image.shape[2] if len(self.image.shape) > 2 else 1
        
        print(f"Image loaded: {image_path}")
        print(f"  - Dimensions: {width}x{height}")
        print(f"  - Channels: {channels}")
        
        return self.image
    
    def convert_to_grayscale(self):
        if self.image is None:
            raise ValueError("No image loaded.")
        
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processing_history.append("grayscale")
            print("Grayscale conversion completed")
        else:
            print("Image already in grayscale")
        
        return self.image
    
    def apply_gaussian_blur(self, kernel_size=5, sigma=0):
        if self.image is None:
            raise ValueError("No image loaded.")
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma)
        self.processing_history.append(f"gaussian_blur_{kernel_size}")
        
        print(f"Gaussian blur applied (kernel={kernel_size}, sigma={sigma})")
        return self.image
    
    def detect_edges_canny(self, threshold1=100, threshold2=200):
        if self.image is None:
            raise ValueError("No image loaded.")
        
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        edges = cv2.Canny(gray, threshold1, threshold2)
        self.processing_history.append(f"canny_{threshold1}_{threshold2}")
        
        print(f"Canny edge detection (seuils={threshold1}, {threshold2})")
        return edges
    
    def detect_edges_sobel(self):
        if self.image is None:
            raise ValueError("No image loaded.")
        
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
        
        self.processing_history.append("sobel")
        print("Sobel edge detection applied")
        return edges
    
    def apply_threshold(self, threshold=127, max_value=255, method='binary'):
        if self.image is None:
            raise ValueError("No image loaded.")
        
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()
        
        methods = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'otsu': cv2.THRESH_BINARY + cv2.THRESH_OTSU
        }
        
        if method == 'adaptive':
            result = cv2.adaptiveThreshold(gray, max_value, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            self.processing_history.append("adaptive_threshold")
        elif method == 'otsu':
            _, result = cv2.threshold(gray, 0, max_value, methods[method])
            self.processing_history.append("otsu_threshold")
        else:
            _, result = cv2.threshold(gray, threshold, max_value, methods.get(method, cv2.THRESH_BINARY))
            self.processing_history.append(f"threshold_{threshold}")
        
        print(f"Threshold applied (méthode={method})")
        return result
    
    def reset_to_original(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.processing_history = []
            print("Image reset to original")
            self.processing_history.append("otsu_threshold")
        else:
            _, result = cv2.threshold(gray, threshold, max_value, methods.get(method, cv2.THRESH_BINARY))
            self.processing_history.append(f"threshold_{threshold}")
        
        print(f" Threshold applied (méthode={method})")
        return result
    
    def reset_to_original(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.processing_history = []
            print("Image reset to original")
