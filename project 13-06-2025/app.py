import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import ast
import re
import os

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Recipe Generator API is running!"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "Server is running!"})

def load_and_preprocess_data():
    try:
        df = pd.read_csv('data_reciepies.csv')
        
        column_mapping = {
            'Title': 'recipe_name',
            'Ingredients': 'ingredients',
            'Instructions': 'instructions'
        }
        df = df.rename(columns=column_mapping)
        
        def convert_ingredients(ing_str):
            try:
                if pd.isna(ing_str):
                    return []
                if isinstance(ing_str, str):
                    ing_str = ing_str.strip('"\'').strip()
                    if ing_str.startswith('[') and ing_str.endswith(']'):
                        return ast.literal_eval(ing_str)
                    return [i.strip() for i in ing_str.split(',')]
                return ing_str if isinstance(ing_str, list) else []
            except:
                return []

        df['ingredients'] = df['ingredients'].apply(convert_ingredients)
        
        def simplify_ingredients(ingredients_list):
            simplified = []
            for ingredient in ingredients_list:
                ingredient = str(ingredient).lower()
                main_ingredient = re.split(r'\d+|\(|\)|\s+cup[s]?|\s+tbsp|\s+tsp|\s+oz|\s+lb[s]?', 
                                        ingredient)[-1].strip()
                main_ingredient = re.sub(r'[^\w\s]', '', main_ingredient)
                main_ingredient = re.sub(r'divided|plus|more|fresh|dried|optional|chopped|minced|diced|sliced', '', main_ingredient).strip()
                if main_ingredient and len(main_ingredient) > 2:
                    simplified.append(main_ingredient)
            return list(set(simplified))
        
        df['simplified_ingredients'] = df['ingredients'].apply(simplify_ingredients)
        df = df[df['simplified_ingredients'].apply(len) > 0]
        df['ingredients_text'] = df['simplified_ingredients'].apply(lambda x: ' '.join(x))
        
        return df
    
    except Exception as e:
        raise

# Load data quietly
df = load_and_preprocess_data()
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
recipe_vectors = tfidf.fit_transform(df['ingredients_text'])

@app.route('/predict', methods=['POST'])
def predict_recipe():
    try:
        data = request.get_json()
        input_ingredients = [ingredient.lower().strip() for ingredient in data['ingredients']]
        
        # Create query vector
        query_text = ' '.join(input_ingredients)
        query_vector = tfidf.transform([query_text])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, recipe_vectors).flatten()
        
        # Get top 5 most similar recipes
        top_indices = similarities.argsort()[-5:][::-1]
        
        recipes = []
        for idx in top_indices:
            if similarities[idx] > 0:
                recipe = df.iloc[idx]
                
                recipe_ingredients = set(recipe['simplified_ingredients'])
                input_ingredients_set = set(input_ingredients)
                
                recipe_coverage = len(recipe_ingredients.intersection(input_ingredients_set)) / len(recipe_ingredients) * 100 if len(recipe_ingredients) > 0 else 0
                input_coverage = len(recipe_ingredients.intersection(input_ingredients_set)) / len(input_ingredients_set) * 100 if len(input_ingredients_set) > 0 else 0
                
                match_percentage = (recipe_coverage + input_coverage) / 2
                
                if match_percentage > 10:
                    recipes.append({
                        'recipe_name': recipe['recipe_name'],
                        'ingredients': recipe['ingredients'],
                        'instructions': recipe['instructions'],
                        'match_percentage': round(match_percentage, 2),
                        'similarity_score': round(similarities[idx] * 100, 2)
                    })
        
        recipes = sorted(recipes, key=lambda x: x['match_percentage'], reverse=True)[:3]
        
        return jsonify({'recipes': recipes})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Recipe Generator API...")
    print("Server running at http://localhost:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)  # Set debug=False to reduce terminal output