import cv2
import numpy as np
import matplotlib.pyplot as plt
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
            raise FileNotFoundError(f"mage not found: {image_path}")
        
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
    def load_image_from_database(self, filename):
        self.db.connect()
        self.db.cursor.execute(
            "SELECT file_path FROM image_metadata WHERE filename = %s", (filename,)
    )
        result = self.db.cursor.fetchone()
        self.db.disconnect()

        if not result:
            raise FileNotFoundError(f"No image found in database with filename: {filename}")

        image_path = result[0]
        return self.load_image(image_path)
    
    def apply_processing_pipeline(self, operations):
        if self.image is None:
            raise ValueError("No image loaded to process.")

        processed = self.image.copy()
        for op, kwargs in operations:
            if op == 'grayscale':
                processed = self.convert_to_grayscale()
            elif op == 'gaussian_blur':
                processed = self.apply_gaussian_blur(**kwargs)
            elif op == 'canny':
                processed = self.detect_edges_canny(**kwargs)
            elif op == 'sobel':
                processed = self.detect_edges_sobel()
            elif op == 'threshold':
                processed = self.apply_threshold(**kwargs)
            else:
                print(f"Unknown operation: {op}")

        return processed

    def display_multiple_processing(self, save_path=None):
  
        if self.original_image is None:
            raise ValueError("No original image loaded.")
    
    #sauvegarder l'image originale pour comparaison
        original = self.original_image.copy()
    
    #appliquer différents traitements
        self.reset_to_original()
        gray = self.convert_to_grayscale()
    
        self.reset_to_original()
        blurred = self.apply_gaussian_blur(kernel_size=7)
    
        self.reset_to_original()
        edges = self.detect_edges_canny(threshold1=50, threshold2=150)
    
        self.reset_to_original()
        threshold_img = self.apply_threshold(method='otsu')
    
        images = [original, gray, blurred, edges, threshold_img]
        titles = ["Original", "Grayscale", "Gaussian Blur", "Canny Edges", "Otsu Threshold"]
    
        fig, axes = plt.subplots(1, len(images), figsize=(18,4))
        for ax, img, title in zip(axes, images, titles):
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
    
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Multiple processing comparison saved: {save_path}")
        plt.show()

    
    def convert_to_grayscale(self):
        
        if self.image is None:
            raise ValueError("No image loaded.")
        
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processing_history.append("grayscale")
            print("Grayscale conversion completed")
        else:
            print(" Image already in grayscale")
        
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
        
        #convertir en niveaux de gris si nécessaire
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
        
        #convertir en niveaux de gris si nécessaire
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
        
        print(f" Threshold applied (méthode={method})")
        return result
    
    def reset_to_original(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.processing_history = []
            print("Image reset to original")
    
    def save_image(self, output_path):

        if self.image is None:
            raise ValueError("No image to save.")
        
        cv2.imwrite(output_path, self.image)
        print(f"Image saved: {output_path}")
    
    def store_metadata(self):
        if self.image is None or self.image_path is None:
            raise ValueError("No image loaded.")
        
        filename = os.path.basename(self.image_path)
        file_size = os.path.getsize(self.image_path)
        height, width = self.original_image.shape[:2]
        methods = ", ".join(self.processing_history) if self.processing_history else "none"
        
        self.db.connect()
        
        self.db.cursor.execute(
            "SELECT id FROM image_metadata WHERE filename = %s", (filename,)
        )
        existing = self.db.cursor.fetchone()
        
        if existing:
            self.db.cursor.execute('''
                UPDATE image_metadata 
                SET processing_methods = %s, updated_at = CURRENT_TIMESTAMP
                WHERE filename = %s
            ''', (methods, filename))
        else:
            self.db.cursor.execute('''
                INSERT INTO image_metadata 
                (filename, file_path, file_size, width, height, processing_methods)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (filename, self.image_path, file_size, width, height, methods))
        
        self.db.connection.commit()
        self.db.disconnect()
        
        print(f"Metadata stored for'{filename}'")
    
    def display_comparison(self, processed_image, title1="Original", title2="Processed", save_path=None): 
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if len(self.original_image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title(title1, fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        if len(processed_image.shape) == 3:
            axes[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(processed_image, cmap='gray')
        axes[1].set_title(title2, fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved: {save_path}")
        
        plt.show()
    
def create_sample_image():
    sample_path = "images/sample_environmental.png"
    
    if not os.path.exists(sample_path):
        #créer une image de test avec des motifs
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        #fond avec gradient (simule le ciel)
        for i in range(400):
            img[i, :] = [255 - i//2, 200 - i//3, 100 + i//4]
        
        #ajouter des formes (simule des zones de pollution)
        cv2.circle(img, (150, 200), 80, (100, 100, 100), -1)
        cv2.circle(img, (450, 150), 60, (80, 80, 80), -1)
        cv2.rectangle(img, (250, 300), (400, 380), (50, 50, 50), -1)
        
        #ajouter du bruit
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(sample_path, img)
        print(f"Sample image created: {sample_path}")
    
    return sample_path


def test_image_processing():

    print("=" * 60)
    print("Image Processing Test")
    print("=" * 60)
    
    processor = ImageProcessor()
    print("\n1. Preparing image..")
    
    images_dir = "images"
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_files:
        image_path = os.path.join(images_dir, image_files[0])
        print(f" Using existing image: {image_path}")
    else:
        image_path = create_sample_image()
    
    print("\n2. Loading image..")
    processor.load_image(image_path)
   
    print("\n3. Converting to grayscale..")
    gray = processor.convert_to_grayscale()
    processor.save_image("images/processed_grayscale.png")
 
    print("\n4. Applying Gaussian blur..")
    processor.reset_to_original()
    blurred = processor.apply_gaussian_blur(kernel_size=7)
    processor.save_image("images/processed_blurred.png")

    print("\n5. Edge detection (Canny)..")
    processor.reset_to_original()
    edges = processor.detect_edges_canny(threshold1=50, threshold2=150)
    cv2.imwrite("images/processed_edges.png", edges)
    print("Image saved: images/processed_edges.png")

    print("\n6. Otsu thresholding..")
    processor.reset_to_original()
    threshold = processor.apply_threshold(method='otsu')
    cv2.imwrite("images/processed_threshold.png", threshold)
    print("Image saved: images/processed_threshold.png")
   
    print("\n7. Storing metadata in the database..")
    processor.reset_to_original()
    processor.convert_to_grayscale()
    processor.apply_gaussian_blur()
    processor.store_metadata()
    
    print("\n8. Creating comparative visualization..")
    processor.reset_to_original()
    processor.display_multiple_processing(save_path="images/image_processing_comparison.png")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_image_processing()
