import random

class MockPredictor:
    def __init__(self):
        self.classes = ["Acne", "Dark Spots", "Normal", "Uneven Texture"]
        
    def predict(self, image_path):
        """
        Simulate prediction. Returns a class name and confidence.
        """
        # Return a random class for demo purposes if model is missing
        predicted_class = random.choice(self.classes)
        confidence = round(random.uniform(0.7, 0.99), 2)
        return predicted_class, confidence
