import csv
import os

class Recommender:
    def __init__(self, product_db_path="data/products.csv"):
        self.product_db_path = product_db_path
        self.products = []
        if os.path.exists(product_db_path):
            with open(product_db_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.products.append(row)
        else:
            print(f"Warning: {product_db_path} not found.")

    def get_recommendations(self, skin_condition, top_k=3):
        """
        Returns top_k products for the given skin condition using simple keyword matching.
        """
        if not self.products:
            return []

        scored_products = []
        condition_lower = skin_condition.lower()

        for prod in self.products:
            score = 0
            target = prod.get('skin_condition_target', '').lower()
            ingredients = prod.get('ingredients', '').lower()
            rating = float(prod.get('rating', 0) or 0)

            # Match condition in target
            if condition_lower in target:
                score += 5
            
            # Match condition in ingredients/description
            if condition_lower in ingredients:
                score += 2
            
            # Add rating weight
            score += (rating / 5.0)

            scored_products.append({
                'prod': prod,
                'score': score
            })

        # Sort by score descending
        scored_products.sort(key=lambda x: x['score'], reverse=True)
        
        recs = [item['prod'] for item in scored_products[:top_k]]
        
        # Add Search Links and clean up
        for rec in recs:
            query = f"{rec.get('brand', '')} {rec.get('product_name', '')}"
            rec['amazon_url'] = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
            rec['flipkart_url'] = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
            
            # Generate a reason
            reasons = []
            target_str = rec.get('skin_condition_target', '').lower()
            ing_str = rec.get('ingredients', '').lower()

            if condition_lower in target_str:
                reasons.append(f"Specifically formulated for {skin_condition}")
            
            if "niacinamide" in ing_str and skin_condition == "Acne":
                reasons.append("Contains Niacinamide to reduce inflammation")
            elif "salicylic acid" in ing_str and skin_condition == "Acne":
                reasons.append("Salicylic acid helps clear pores")
            elif "vitamin c" in ing_str and "Dark Spots" in skin_condition:
                reasons.append("Vitamin C brightens and fades dark spots")
            elif "hyaluronic" in ing_str or "glycerin" in ing_str:
                reasons.append("Provides deep hydration")
            
            rec['reason'] = ". ".join(reasons[:2]) if reasons else "Highly rated for your skin type"

        return recs

if __name__ == "__main__":
    rec = Recommender()
    print(f"Recommendations for 'Acne':")
    for r in rec.get_recommendations("Acne"):
        print(f"- {r.get('product_name')} ({r.get('brand')})")
