import os
import numpy as np
import cv2
from pathlib import Path

# Configuration
DATA_DIR = Path("data/raw")
CLASSES = ["Acne", "Dark Spots", "Normal", "Uneven Texture"]
IMG_SIZE = (224, 224)
NUM_SAMPLES = 100  # Increased for better training

def create_synthetic_data():
    """
    Generates realistic looking synthetic skin images with specific conditions.
    """
    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating data for {class_name}...")
        
        for i in range(NUM_SAMPLES):
            # 1. Base skin tone (varied across samples)
            # Ranges from fair to deep
            base_r = np.random.randint(180, 255)
            base_g = np.random.randint(140, 210)
            base_b = np.random.randint(110, 180)
            
            img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
            img[:, :, 2] = base_r # R
            img[:, :, 1] = base_g # G
            img[:, :, 0] = base_b # B
            
            # Add subtle skin texture (uniform noise)
            texture = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)

            # 2. Add class-specific features
            if class_name == "Acne":
                # Red clusters with white/yellow centers (pimples)
                num_spots = np.random.randint(5, 15)
                for _ in range(num_spots):
                    x, y = np.random.randint(20, 204, 2)
                    size = np.random.randint(3, 7)
                    # Red halo
                    cv2.circle(img, (x, y), size + 2, (30, 30, 200), -1)
                    # Lighter center
                    cv2.circle(img, (x, y), size // 2, (150, 200, 255), -1)
                
                # Apply Gaussian blur to make it look "under the skin"
                img = cv2.GaussianBlur(img, (3, 3), 0)
            
            elif class_name == "Dark Spots":
                # Brownish/Darker pigment patches (hyperpigmentation)
                num_patches = np.random.randint(3, 8)
                overlay = img.copy()
                for _ in range(num_patches):
                    x, y = np.random.randint(20, 204, 2)
                    axes = (np.random.randint(10, 30), np.random.randint(5, 15))
                    angle = np.random.randint(0, 180)
                    # Dark brown color
                    cv2.ellipse(overlay, (x, y), axes, angle, 0, 360, (20, 40, 80), -1)
                
                # Blend for realism
                alpha = 0.4
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                img = cv2.GaussianBlur(img, (5, 5), 0)
            
            elif class_name == "Uneven Texture":
                # High-frequency noise and "pores"
                overlay = img.copy()
                for _ in range(200):
                    x, y = np.random.randint(0, 224, 2)
                    cv2.circle(overlay, (x, y), 1, (20, 20, 20), -1)
                
                alpha = 0.3
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                # Speckle noise
                gauss = np.random.normal(0, 15, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

            elif class_name == "Normal":
                # Just slight variations and smooth blur
                img = cv2.GaussianBlur(img, (3, 3), 0)

            # Save
            cv2.imwrite(str(class_dir / f"{class_name}_{i}.jpg"), img)
            
    print(f"Synthetic data generation complete. {NUM_SAMPLES * len(CLASSES)} images created.")

if __name__ == "__main__":
    create_synthetic_data()
