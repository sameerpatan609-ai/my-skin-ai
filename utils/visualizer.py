import cv2
import numpy as np

def detect_hotspots(img, condition):
    """
    Highlights regions likely containing the specified condition.
    Returns: Annotated image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = None
    color = (0, 0, 255) # Default red
    
    if condition == "Acne":
        # Look for red-ish spots
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        color = (0, 0, 255) # Red for acne

    elif condition == "Dark Spots":
        # Look for brownish/darker spots
        # Brown is roughly low H, moderate S, low V
        lower_brown = np.array([0, 30, 20])
        upper_brown = np.array([30, 150, 100])
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        color = (0, 69, 139) # SaddleBrown for dark spots

    elif condition == "Uneven Texture":
        # Highlight high-contrast areas (using Canny edge)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        mask = cv2.dilate(edges, None, iterations=2)
        color = (255, 0, 255) # Magenta for texture

    if mask is not None:
        # Filter noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output = img.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10: # Minimum area to consider
                x, y, w, h = cv2.boundingRect(cnt)
                # Draw rounded rectangle (simulated) or just box
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                
        return output
    
    return img
