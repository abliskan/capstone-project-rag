# Step to run the tests on your local computer
1. Install pytest (if not already installed)
```
bashpip install pytest pytest-mock
```
2. Save the test file as test_extract_knowledge_data.py in the same directory as your main script
3. Run the tests
```
# Run all tests
pytest test_extract_knowledge_data.py -v

# Run specific test class
pytest test_extract_knowledge_data.py::TestGetNutritionNinjaData -v

# Run with coverage report
pip install pytest-cov
pytest test_extract_knowledge_data.py --cov=extract_knowledge_data --cov-report=html
```