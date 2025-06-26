import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import requests

# Import the functions to test
from extract_knowledge_data import (
    get_nutrition_ninja_data,
    get_nutritionix_data,
    get_recipe_data,
    clean_recipe_data,
    save_to_csv,
    main
)


class TestGetNutritionNinjaData:
    """
    Test cases for get_nutrition_ninja_data function
    """
    
    @patch('extract_knowledge_data.requests.get')
    @patch('extract_knowledge_data.sleep')
    def test_successful_api_call(self, mock_sleep, mock_get):
        """
        Test successful API call with valid response
        """
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "name": "apple",
                "serving_size_g": 100,
                "calories": 52,
                "protein_g": 0.3,
                "carbohydrates_total_g": 14,
                "fat_total_g": 0.2,
                "fiber_g": 2.4
            }
        ]
        mock_get.return_value = mock_response
        
        result = get_nutrition_ninja_data(["1 apple"])
        
        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]["name"] == "apple"
        assert result.iloc[0]["calories"] == 52
        mock_get.assert_called_once()
        mock_sleep.assert_called_once_with(1)
    
    @patch('extract_knowledge_data.requests.get')
    @patch('extract_knowledge_data.sleep')
    def test_empty_response(self, mock_get):
        """Test API call with empty response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        result = get_nutrition_ninja_data(["nonexistent food"])
        
        assert result.empty
    
    @patch('extract_knowledge_data.requests.get')
    @patch('extract_knowledge_data.sleep')
    def test_api_error_response(self, mock_get):
        """
        Test API call with error status code
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = get_nutrition_ninja_data(["invalid query"])
        
        assert result.empty
    
    @patch('extract_knowledge_data.requests.get')
    @patch('extract_knowledge_data.sleep')
    def test_request_exception(self, mock_get):
        """Test handling of request exceptions"""
        mock_get.side_effect = requests.RequestException("Connection error")
        
        result = get_nutrition_ninja_data(["test food"])
        
        assert result.empty

class TestGetNutritionixData:
    """
    Test cases for get_nutritionix_data function
    """
    
    @patch('extract_knowledge_data.requests.post')
    @patch('extract_knowledge_data.sleep')
    def test_successful_api_call(self, mock_post):
        """
        Test successful Nutritionix API call
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "foods": [
                {
                    "food_name": "apple",
                    "serving_weight_grams": 100,
                    "nf_calories": 52,
                    "nf_protein": 0.3,
                    "nf_total_carbohydrate": 14,
                    "nf_total_fat": 0.2,
                    "nf_dietary_fiber": 2.4
                }
            ]
        }
        mock_post.return_value = mock_response
        
        result = get_nutritionix_data(["1 apple"])
        
        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]["name"] == "apple"
        assert result.iloc[0]["calories"] == 52
    
    @patch('extract_knowledge_data.requests.post')
    @patch('extract_knowledge_data.sleep')
    def test_empty_foods_response(self, mock_sleep, mock_post):
        """Test response with no foods"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"foods": []}
        mock_post.return_value = mock_response
        
        result = get_nutritionix_data(["invalid food"])
        
        assert result.empty
    
    @patch('extract_knowledge_data.requests.post')
    @patch('extract_knowledge_data.sleep')
    def test_api_error(self, mock_sleep, mock_post):
        """
        Test API error handling
        """
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        
        result = get_nutritionix_data(["test food"])
        
        assert result.empty


class TestGetRecipeData:
    """
    Test cases for get_recipe_data function
    """
    
    @patch('extract_knowledge_data.requests.get')
    @patch('extract_knowledge_data.sleep')
    def test_successful_recipe_fetch(self, mock_sleep, mock_get):
        """
        Test successful recipe data fetching
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meals": [
                {
                    "strMeal": "Chicken Curry",
                    "strArea": "Indian",
                    "strCategory": "Chicken",
                    "strInstructions": "Cook chicken with spices",
                    "strTags": "Spicy,Curry",
                    "strIngredient1": "Chicken",
                    "strMeasure1": "500g",
                    "strIngredient2": "Onion",
                    "strMeasure2": "1 large",
                    "strIngredient3": "",  # Empty ingredient to test filtering
                    "strMeasure3": ""
                }
            ]
        }
        mock_get.return_value = mock_response
        
        df_recipe, df_ingredient = get_recipe_data(["chicken"])
        
        assert not df_recipe.empty
        assert not df_ingredient.empty
        assert len(df_recipe) == 1
        assert df_recipe.iloc[0]["title"] == "Chicken Curry"
        assert "Chicken" in df_ingredient["name"].values
        assert "Onion" in df_ingredient["name"].values
    
    @patch('extract_knowledge_data.requests.get')
    @patch('extract_knowledge_data.sleep')
    def test_no_meals_found(self, mock_sleep, mock_get):
        """
        Test when no meals are found
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"meals": None}
        mock_get.return_value = mock_response
        
        df_recipe, df_ingredient = get_recipe_data(["nonexistent"])
        
        assert df_recipe.empty
        assert df_ingredient.empty
    
    @patch('extract_knowledge_data.requests.get')
    @patch('extract_knowledge_data.sleep')
    def test_api_error(self, mock_sleep, mock_get):
        """
        Test API error handling
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        df_recipe, df_ingredient = get_recipe_data(["test"])
        
        assert df_recipe.empty
        assert df_ingredient.empty


class TestCleanRecipeData:
    """
    Test cases for clean_recipe_data function
    """
    
    def test_clean_empty_dataframe(self):
        """
        Test cleaning empty DataFrame
        """
        df_empty = pd.DataFrame()
        result = clean_recipe_data(df_empty)
        assert result.empty
    
    def test_fill_missing_nutrition(self):
        """
        Test filling missing nutrition values
        """
        df_recipe = pd.DataFrame({
            "title": ["Recipe 1", "Recipe 2"],
            "ingredients": ["ingredient1", "ingredient2"],
            "instructions": ["step1", "step2"],
            "nutrition": [None, ""]
        })
        
        result = clean_recipe_data(df_recipe)
        
        assert not pd.isna(result.iloc[0]["nutrition"])
        assert not pd.isna(result.iloc[1]["nutrition"])
        assert isinstance(result.iloc[0]["nutrition"], (int, np.integer))


class TestMainFunction:
    """
    Test cases for main function
    """
    
    @patch('extract_knowledge_data.get_nutrition_ninja_data')
    @patch('extract_knowledge_data.get_nutritionix_data')
    @patch('extract_knowledge_data.get_recipe_data')
    @patch('extract_knowledge_data.clean_recipe_data')
    @patch('extract_knowledge_data.save_to_csv')
    def test_main_execution_flow(self, mock_save, mock_clean, mock_recipe, 
                                mock_nutritionix, mock_ninja):
        """
        Test the main function execution flow
        """
        # Mock return values
        mock_ninja.return_value = pd.DataFrame({"name": ["apple"], "calories": [52]})
        mock_nutritionix.return_value = pd.DataFrame({"name": ["banana"], "calories": [89]})
        mock_recipe.return_value = (
            pd.DataFrame({"title": ["Recipe"]}),
            pd.DataFrame({"name": ["ingredient"]})
        )
        mock_clean.return_value = pd.DataFrame({"title": ["Clean Recipe"]})
        
        # Run main function
        main()
        
        # Verify all functions were called
        mock_ninja.assert_called_once()
        mock_nutritionix.assert_called_once()
        mock_recipe.assert_called_once()
        mock_clean.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('extract_knowledge_data.get_nutrition_ninja_data')
    @patch('extract_knowledge_data.get_nutritionix_data')
    @patch('extract_knowledge_data.get_recipe_data')
    @patch('extract_knowledge_data.clean_recipe_data')
    @patch('extract_knowledge_data.save_to_csv')
    def test_main_with_empty_nutrition_data(self, mock_save, mock_clean, mock_recipe, 
                                           mock_nutritionix, mock_ninja):
        """
        Test main function when nutrition APIs return empty data
        """
        # Mock empty nutrition data
        mock_ninja.return_value = pd.DataFrame()
        mock_nutritionix.return_value = pd.DataFrame()
        mock_recipe.return_value = (
            pd.DataFrame({"title": ["Recipe"]}),
            pd.DataFrame({"name": ["ingredient"]})
        )
        mock_clean.return_value = pd.DataFrame({"title": ["Clean Recipe"]})
        
        # Should not raise exception
        main()
        
        mock_save.assert_called_once()
    
    @patch('extract_knowledge_data.get_nutrition_ninja_data')
    def test_main_exception_handling(self, mock_ninja):
        """
        Test main function exception handling
        """
        mock_ninja.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            main()


# Fixtures for common test data
@pytest.fixture
def sample_nutrition_data():
    """
    Sample nutrition data for testing
    """
    return pd.DataFrame({
        "name": ["apple", "banana", "chicken breast"],
        "serving_size_g": [100, 120, 100],
        "calories": [52, 89, 165],
        "protein_g": [0.3, 1.1, 31],
        "carbohydrates_total_g": [14, 23, 0],
        "fat_total_g": [0.2, 0.3, 3.6],
        "fiber_g": [2.4, 2.6, 0]
    })


@pytest.fixture
def sample_recipe_data():
    """
    Sample recipe data for testing
    """
    return pd.DataFrame({
        "title": ["Chicken Curry", "Apple Pie"],
        "description": ["Spicy chicken dish", "Sweet apple dessert"],
        "servings": [4, 8],
        "cook_time_minutes": [30, 60],
        "ingredients": ["chicken, onion, spices", "apples, flour, sugar"],
        "instructions": ["Cook chicken with spices", "Bake apples in pastry"],
        "nutrition": [400, 250],
        "tags": ["spicy", "sweet"]
    })


@pytest.fixture
def sample_ingredient_data():
    """
    Sample ingredient data for testing
    """
    return pd.DataFrame({
        "name": ["chicken", "onion", "apple", "flour"],
        "quantity": ["500g", "1 large", "3 medium", "2 cups"],
        "category": ["meat", "vegetable", "fruit", "baking"]
    })


# Integration-style tests using fixtures
class TestIntegrationWithFixtures:
    """
    Integration tests using sample data fixtures
    """
    
    def test_nutrition_data_structure(self, sample_nutrition_data):
        """
        Test that nutrition data has expected structure
        """
        required_columns = ["name", "calories", "protein_g"]
        for col in required_columns:
            assert col in sample_nutrition_data.columns
        
        assert len(sample_nutrition_data) > 0
        assert sample_nutrition_data["calories"].dtype in [np.int64, np.float64]
    
    def test_recipe_data_cleaning(self, sample_recipe_data):
        """
        Test recipe data cleaning with sample data
        """
        # Add some dirty data
        dirty_data = sample_recipe_data.copy()
        dirty_data.loc[len(dirty_data)] = {
            "title": "",
            "description": "Empty title recipe",
            "servings": 1,
            "cook_time_minutes": 15,
            "ingredients": "some ingredients",
            "instructions": "some instructions",
            "nutrition": None,
            "tags": "test"
        }
        
        cleaned = clean_recipe_data(dirty_data)
        
        # Should remove the row with empty title
        assert len(cleaned) == len(sample_recipe_data)
        # Should fill nutrition values
        assert not cleaned["nutrition"].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])