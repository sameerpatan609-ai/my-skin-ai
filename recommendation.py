import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class Recommender:
    def __init__(self, product_db_path="data/products.csv"):
        self.product_db_path = product_db_path
        if os.path.exists(product_db_path):
            self.df = pd.read_csv(product_db_path)
            # Fill NaN values
            self.df['ingredients'] = self.df['ingredients'].fillna('')
            self.df['skin_condition_target'] = self.df['skin_condition_target'].fillna('')
            
            # Combine features for content-based filtering
            self.df['features'] = self.df['skin_condition_target'] + " " + self.df['ingredients'] + " " + self.df['category']
            
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['features'])
        else:
            print(f"Warning: {product_db_path} not found.")
            self.df = pd.DataFrame()

    def get_recommendations(self, skin_condition, top_k=3):
        """
        Returns top_k products for the given skin condition using TF-IDF similarity.
        """
        if self.df.empty:
            return []

        # Vectorize the input condition
        condition_vec = self.vectorizer.transform([skin_condition])
        
        # Calculate cosine similarity between condition and all products
        cosine_sim = cosine_similarity(condition_vec, self.tfidf_matrix).flatten()
        
        # Get the top K indices, but also consider rating a bit
        # Final score = 0.7 * similarity + 0.3 * (rating / 5)
        # Handle cases where rating might be missing
        ratings = self.df['rating'].fillna(0).values / 5.0
        final_scores = 0.7 * cosine_sim + 0.3 * ratings
        
        top_indices = final_scores.argsort()[-top_k:][::-2] # Reverse order
        # Actually let's just use argsort and take last K
        top_indices = final_scores.argsort()[-top_k:][::-1]
        
        recs = self.df.iloc[top_indices].copy().to_dict('records')
        
        # Add Search Links and clean up
        for rec in recs:
            query = f"{rec['brand']} {rec['product_name']}"
            rec['amazon_url'] = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
            rec['flipkart_url'] = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
            
            # Generate a reason
            reasons = []
            if skin_condition.lower() in rec['skin_condition_target'].lower():
                reasons.append(f"Specifically formulated for {skin_condition}")
            
            if "niacinamide" in rec['ingredients'].lower() and skin_condition == "Acne":
                reasons.append("Contains Niacinamide to reduce inflammation")
            elif "salicylic acid" in rec['ingredients'].lower() and skin_condition == "Acne":
                reasons.append("Salicylic acid helps clear pores")
            elif "vitamin c" in rec['ingredients'].lower() and "Dark Spots" in skin_condition:
                reasons.append("Vitamin C brightens and fades dark spots")
            elif "hyaluronic" in rec['ingredients'].lower() or "glycerin" in rec['ingredients'].lower():
                reasons.append("Provides deep hydration")
            
            rec['reason'] = ". ".join(reasons[:2]) if reasons else "Highly rated for your skin type"

            # Remove internal features column from dict
            if 'features' in rec:
                del rec['features']
                
        return recs

if __name__ == "__main__":
    rec = Recommender()
    print(f"Recommendations for 'Acne':")
    for r in rec.get_recommendations("Acne"):
        print(f"- {r['product_name']} ({r['brand']})")

