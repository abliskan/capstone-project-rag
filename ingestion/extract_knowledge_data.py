import requests
import pandas as pd
import numpy as np
import os
from time import sleep
from typing import List

# API CREDENTIALS
NINJA_API_KEY = os.getenv("NINJA_API_KEY")
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_APP_KEY = os.getenv("NUTRITIONIX_APP_KEY")

# Fetch nutrition data from API Ninja
def get_nutrition_ninja_data(food_queries: List[str]) -> pd.DataFrame:
    """
    Fetch nutrition data from API Ninja
    """
    print("Fetching data from Nutrition API Ninja...")
    nutrition_ninja = []
    
    for i, query in enumerate(food_queries, 1):
        print(f"Processing {i}/{len(food_queries)}: {query}")
        try:
            r = requests.get(
                "https://api.api-ninjas.com/v1/nutrition",
                headers={"X-Api-Key": NINJA_API_KEY},
                params={"query": query},
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                if data:  # Check if response is not empty
                    nutrition_ninja.extend(data)
                else:
                    print(f"No data found for: {query}")
            else:
                print(f"Error {r.status_code} for: {query}")
        except requests.RequestException as e:
            print(f"Request failed for {query}: {e}")
        
        sleep(1)  # Rate limiting
    
    if nutrition_ninja:
        df_ninja = pd.DataFrame(nutrition_ninja)
        # Select only the columns we need if they exist
        available_cols = [col for col in ["name", "serving_size_g", "calories", "protein_g", "carbohydrates_total_g", "fat_total_g", "fiber_g"] 
                         if col in df_ninja.columns]
        return df_ninja[available_cols]
    else:
        print("No data retrieved from Nutrition API Ninja")
        return pd.DataFrame()

# Fetch nutrition data from Nutritionix API
def get_nutritionix_data(food_queries: List[str]) -> pd.DataFrame:
    """
    Fetch nutrition data from Nutritionix API
    """
    print("\nFetching data from Nutritionix API...")
    nutritionix_records = []
    
    for i, query in enumerate(food_queries, 1):
        print(f"Processing {i}/{len(food_queries)}: {query}")
        try:
            r = requests.post(
                "https://trackapi.nutritionix.com/v2/natural/nutrients",
                headers={
                    "x-app-id": NUTRITIONIX_APP_ID,
                    "x-app-key": NUTRITIONIX_APP_KEY,
                    "Content-Type": "application/json"
                },
                json={"query": query},
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                for f in data.get("foods", []):
                    nutritionix_records.append({
                        "name": f.get("food_name"),
                        "serving_size_g": f.get("serving_weight_grams"),
                        "calories": f.get("nf_calories"),
                        "protein_g": f.get("nf_protein"),
                        "carbohydrates_total_g": f.get("nf_total_carbohydrate"),
                        "fat_total_g": f.get("nf_total_fat"),
                        "fiber_g": f.get("nf_dietary_fiber")
                    })
            else:
                print(f"Error {r.status_code} for: {query}")
        except requests.RequestException as e:
            print(f"Request failed for {query}: {e}")
        
        sleep(1)  # Rate limiting

    if nutritionix_records:
        return pd.DataFrame(nutritionix_records)
    else:
        print("No data retrieved from Nutritionix API")
        return pd.DataFrame()

# Fetch recipe and ingredient data from The MealDB
def get_recipe_data(keywords: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch recipe and ingredient data from TheMealDB
    """
    print("\nFetching data from TheMealDB...")
    recipe_list = []
    ingredient_list = []

    for i, keyword in enumerate(keywords, 1):
        print(f"Processing keyword {i}/{len(keywords)}: {keyword}")
        try:
            r = requests.get(
                f"https://www.themealdb.com/api/json/v1/1/search.php?s={keyword}",
                timeout=10
            )
            if r.status_code == 200:
                meals = r.json().get("meals", [])
                if meals:
                    for meal in meals:
                        # Extract recipe information
                        recipe_list.append({
                            "title": meal.get("strMeal"),
                            "description": f"{meal.get('strMeal')} from {meal.get('strArea')} - {meal.get('strCategory')}",
                            "servings": 1,
                            "cook_time_minutes": None,
                            "ingredients": "; ".join([
                                f"{meal.get(f'strMeasure{i}', '').strip()} {meal.get(f'strIngredient{i}', '').strip()}"
                                for i in range(1, 21)
                                if meal.get(f'strIngredient{i}') and meal.get(f'strIngredient{i}').strip()
                            ]),
                            "instructions": meal.get("strInstructions"),
                            "nutrition": meal.get("nutrition"),
                            "tags": meal.get("strTags")
                        })

                        # Extract ingredients
                        for j in range(1, 21):
                            ing = meal.get(f"strIngredient{j}")
                            qty = meal.get(f"strMeasure{j}")
                            if ing and ing.strip():
                                ingredient_list.append({
                                    "name": ing.strip(),
                                    "quantity": qty.strip() if qty else "",
                                    "category": meal.get("strCategory")
                                })
                else:
                    print(f"No meals found for: {keyword}")
            else:
                print(f"Error {r.status_code} for: {keyword}")
        except requests.RequestException as e:
            print(f"Request failed for {keyword}: {e}")
        
        sleep(1)  # Rate limiting

    # Create DataFrames
    df_recipe = pd.DataFrame(recipe_list).drop_duplicates().head(100) if recipe_list else pd.DataFrame()
    df_ingredient = pd.DataFrame(ingredient_list).drop_duplicates().head(100) if ingredient_list else pd.DataFrame()

    return df_recipe, df_ingredient

# Clean and validate recipe data
def clean_recipe_data(df_recipe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate recipe data
    """
    if df_recipe.empty:
        return df_recipe
    
    # Fill missing nutrition values with random numbers (as in original code)
    df_recipe["nutrition"] = df_recipe["nutrition"].apply(
        lambda x: np.random.randint(1, 1001) if pd.isna(x) or str(x).strip() == "" else x
    )
    
    # Remove rows with missing essential data
    df_recipe.dropna(subset=["title", "ingredients", "instructions"], inplace=True)
    df_recipe = df_recipe[
        (df_recipe["title"].str.strip() != "") &
        (df_recipe["ingredients"].str.strip() != "") &
        (df_recipe["instructions"].str.strip() != "")
    ].reset_index(drop=True)
    
    return df_recipe

def save_to_csv(df_nutrition: pd.DataFrame, df_ingredient: pd.DataFrame, df_recipe: pd.DataFrame):
    """
    Save DataFrames to CSV files
    """
    print("\nSaving data to CSV files...")
    
    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each DataFrame to a separate CSV file
    if not df_nutrition.empty:
        nutrition_file = os.path.join(output_dir, "External_knowledge_NutritionInfo.csv")
        df_nutrition.to_csv(nutrition_file, index=False)
        print(f"Nutrition data saved to: {nutrition_file} ({len(df_nutrition)} records)")
    else:
        print("No nutrition data to save")
    
    if not df_ingredient.empty:
        ingredient_file = os.path.join(output_dir, "External_knowledge_Ingredient.csv")
        df_ingredient.to_csv(ingredient_file, index=False)
        print(f"Ingredient data saved to: {ingredient_file} ({len(df_ingredient)} records)")
    else:
        print("No ingredient data to save")
    
    if not df_recipe.empty:
        recipe_file = os.path.join(output_dir, "External_knowledge_Recipe.csv")
        df_recipe.to_csv(recipe_file, index=False)
        print(f"Recipe data saved to: {recipe_file} ({len(df_recipe)} records)")
    else:
        print("No recipe data to save")

def main():
    print("Starting nutrition data extraction...")
    # Sample food queries
    food_queries = [
        "1 apple", "1 banana", "1 cup rice", "2 eggs", "1 avocado",
        "100g chicken breast", "1 slice bread", "1 cup milk", "1 tbsp peanut butter",
        "1 cup broccoli", "1 orange", "1 cup coffee", "1 cup spinach", "1 cup yogurt"
    ]
    
    # Recipe search keywords
    recipe_keywords = ["chicken", "beef", "rice", "vegetable", "egg", "pasta"]
    
    try:
        # 1. Get nutrition data from APIs
        df_ninja = get_nutrition_ninja_data(food_queries)
        df_nutritionix = get_nutritionix_data(food_queries)
        
        # Combine nutrition data
        nutrition_dfs = [df for df in [df_ninja, df_nutritionix] if not df.empty]
        if nutrition_dfs:
            df_nutrition = pd.concat(nutrition_dfs, ignore_index=True).drop_duplicates().head(100)
        else:
            df_nutrition = pd.DataFrame()
        
        # 2. Get recipe and ingredient data
        df_recipe, df_ingredient = get_recipe_data(recipe_keywords)
        
        # 3. Clean recipe data
        df_recipe = clean_recipe_data(df_recipe)
        
        # 4. Save to CSV files
        save_to_csv(df_nutrition, df_ingredient, df_recipe)
        
        print("\n=== Data Extraction Complete ===")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()