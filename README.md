# SourChef – AI-Powered Nutrition & Meal Recommender

**SourChef** is an intelligent, RAG-enhanced chatbot designed for nutrition-focused food recommendations, personalized meal planning, dietary filtering, calorie/macronutrient tracking, and YouTube video recipe suggestions. Built with Gemini Pro (via LangChain), Pinecone, Langfuse, Streamlit, and SQLite.

## Problem Definition
- Information Overload & Confusion: 68% of people report being confused about nutrition advice due to conflicting information
- Lack of Personalization: people often find that generic "one-size-fits-all" nutrition advice, like following a fixed calorie count or macronutrient ratio, doesn't work for them
- Recipe-Nutrition Disconnect: people can't easily connect recipes to their nutritional goals

## Key Features
- **Conversational Intelligence**: User can get recommends personalized, health-aligned food and recipes using memory-augmented retrieval and reasoning.
- **Retrieval-Augmented Generation (RAG) Implementation** from vectorized food, nutrition, and recipe knowledge base via Pinecone vectorstore.
- **Dietary filtering**: Chatbot can act like a helpful "personal diet planner" (e.g. vegan, keto, gluten-free).
- **Calorie & macronutrient calculator**: Chabot use nutrition and calorie from external source to prepare your nutrition and recipe data before jumping into recommendation.
- **YouTube recipe video suggestions**: Videos displayed alongside recipe responses.
- **LLM monitoring**: Traces, tags, and logs every user query and LLM response.
- 📋 **Mutli-Tab Streamlit UI (using menu)**:
  - 💬 Chat: Main page to interact with the AI/LLM
  - 🍽️ Weekly Meal Planner: Meal Plan: Auto-generate a weekly plan from saved recipes. Builds 7-day x 3-meal (breakfast, lunch, and dinner) plan using saved favorite recipes
  - ⭐ Favorite Recipes: User favorite recipes or past chat with AI/LLM
  - 📊 Analytics Dashboard: Dashboard page to see visualization of user behavior and user–chatbot interactions

## Project Structure
Sourchef-Bot
├── main.py                  # Contains the streamlit code for the LLM RAG Chatbot
├── .venv/                   # Not show in this repo - manage python package and other
├── data/                    # External data source
│   ├── Combined_Nutrition_Data.xlsx
│   ├── External_knowledge_Ingredient.csv
│   ├── External_knowledge_NutritionInfo.csv
│   └── External_knowledge_Recipe.csv
│
├── chains/                  # RAG core logic
│   ├── rag_chains.py  
|
├── notebooks/               # Notebook for exploration
│   ├── 01_Data-ingestion-for-rag-chatbot.ipynb
│   ├── 02_1_Pinecone_cohore_integration_connector_1.ipynb
│   ├── 02_2_Pinecone_cohore_integration_connector_2.ipynb
│   ├── 02_3_Pinecone_cohore_integration_connector_3.ipynb
│   └── Retrieval-QA-Pipeline-in-LangChain.ipynb
│
├── assets/                   # Icons, CSS for Streamlit customization, and screenshot
│
├── tests/                    # Tests
│   ├── README.md
│   └── test_extract_knowledge_data.py
|
├── .env                      # Secrets for Gemini, Pinecone, Langfuse, etc...
├── .gitignore
├── .python-version
├── requirements.txt          # Python dependencies
├── pyproject
└── README.md

## Project Architecture

![architecture](https://github.com/abliskan/capstone-project-data-science-1/blob/main/asset/Chatbot-workflow.png)


## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.12+
- **Embedding model**: Gemini API/models/embedding-001
- **LLM model**: Gemini API/models/gemini-2.0-flash
- **AI Framework**: LangChain
- **Vector Database**: Pinecone Cloud
- **LLM Monitoring**: Langfuse Cloud
- **Chat Memory**: ConversationBufferMemory
- **Containerization**: Docker
- **Deployment**: Streamlit cloud
- **CI/CD Pipeline**: Github actions
- **Package Manager**: uv
- **Python Linting**: ruff

## Getting Started and Running

### 1. Get the API key 
- Obtain API key from [Google-AI-Studio](https://aistudio.google.com/app/apikey)
- Obtain API key from [Pinecone-key](https://docs.pinecone.io/guides/projects/manage-api-keys)
- Obtain API key from [YouTube-Data-API-v3](https://developers.google.com/youtube/registering_an_application)
- Obtain API key from [Nutritionix-API](https://www.nutritionix.com/business/api)
- Obtain API key from [Langfuse-API](https://langfuse.com/faq/all/where-are-langfuse-api-keys)

### 2. Clone the repo

```bash
git git@github.com:abliskan/capstone-project-data-science-1.git
cd capstone-project-data-science-1
```

### 3. Install dependencies (via uv or pip)
```bash
conda create --name <environment_name> python=<python_version>
conda activate <env_name>
pip install uv
uv venv
.venv\Scripts\activate
uv pip install python-dotenv langchain langchain-community langchain-pinecone langchain-google-genai langchainhub langfuse pandas streamlit streamlit-extras streamlit-chat requests google-api-python-client
uv pip install --dev pytest
uv pip list
```

### 3. Change your .env file
```
touch .env

# insert the API Key
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
PINECONE_INDEX_NAME=your_pinecone_index
YOUTUBE_API_KEY=your_serpapi_key
NUTRITIONIX_APP_ID=your_nutritionix_app_id
NUTRITIONIX_API_KEY=your_nutritionix_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://app.langfuse.com
```

### 4. Run the App 

Run the streamlit from your local computer
```
streamlit run main.py
```

## Future Improvements
- Multilingual Q&A support [example: Bahasa Indonesia, Japanese(Kanji), Korean(Hangul), etc.)
- Export favorite meals as grocery list (PDF)
- Integrate voice chat via Whisper/Speech-to-Text
- Allow exporting weekly plans and nutrition breakdown in PDF/CSV
- Use multimodal Support to add image input (e.g. food photos)
- Use Auth for manage multiple users with real login feature

## Project Screenshot

### Sourchef Bot - Streamlit UI
![Bot-UI](https://github.com/vnobets7/final_project_ftde_ricky/blob/main/Data-visualization-with-looker-studio/images/SS-System-architecture.PNG)

### LLM Monitoring - Langfuse
![Langfuse-Dashboard](https://github.com/vnobets7/final_project_ftde_ricky/blob/main/Data-visualization-with-looker-studio/images/SS-BI-Dashboard.PNG)

## Diclaimer
This is just experiment project and wasn't a substitute of perfessional nutrition expert or health organization.
