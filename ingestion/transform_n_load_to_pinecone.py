import pandas as pd
import numpy as np
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_google_genai import GoogleGenerativeAIEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone  # <--- THIS is LangChain's wrapper
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import json
import getpass
import os
from time import sleep

def row_to_text(row):
        sheet = str(row.get("__sheet__", "")).strip()
        if sheet == "NutritionInfo":
            return (
                f"Nutrition facts for {row.get('name', 'unknown')}: "
                f"{row.get('calories', 'N/A')} calories, "
                f"{row.get('protein_g', 'N/A')}g protein, "
                f"{row.get('carbohydrates_total_g', 'N/A')}g carbohydrates, "
                f"{row.get('fat_total_g', 'N/A')}g fat, "
                f"{row.get('fiber_g', 'N/A')}g fiber."
            )

        elif sheet == "Ingredient":
            return (
                f"Ingredient: {row.get('name', 'unknown')}. "
                f"Quantity: {row.get('quantity', 'unspecified')}. "
            )

        elif sheet == "Recipe":
            return (
                f"Recipe Title: {row.get('title', 'Untitled')}. "
                f"Description: {row.get('description', '')}. "
                f"Ingredients: {row.get('ingredients', '')}. "
                f"Instructions: {row.get('instructions', '')}. "
                f"Nutrition Score: {row.get('nutrition', 'unknown')}. "
            )

        return "Uncategorized entry."
    # Apply to all rows
    texts = df.apply(row_to_text, axis=1).tolist()
    metadatas = df.to_dict(orient="records")
    texts = df_ingredient.apply(row_to_text, axis=1).tolist()
    metadatas = df_ingredient.to_dict(orient="records")
    texts = df_recipe.apply(row_to_text, axis=1).tolist()
    metadatas = df_recipe.to_dict(orient="records")

    print(f"Converted {len(texts)} rows to documents.")

def main():
    # Pinecone AUTH
    api_key_pinecone = getpass.getpass("Input API KEY PINECONE CLOUD")
    print('api_key_pinecone telah diinput')

    df_nutrition = pd.read_csv("/content/External_knowledge_NutritionInfo.csv")
    df_nutrition = df_nutrition.dropna()
    df_ingredient = pd.read_csv("/content/External_knowledge_Ingredient.csv")
    df_ingredient.head()
    df_recipe = pd.read_csv("/content/External_knowledge_Recipe.csv")
    df_recipe.head()

    # Add sheet label
    df_ingredient["__sheet__"] = "Ingredient"    
    df_nutrition["__sheet__"] = "NutritionInfo"
    df_recipe["__sheet__"] = "Recipe"

    # Gemini AI AUTH
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = []
    chunk_metadata = []

    for i, text in enumerate(texts):
        splits = text_splitter.split_text(text)
        chunks.extend(splits)
        chunk_metadata.extend([metadatas[i]] * len(splits))
    
    embeddings = embedding_model.embed_documents(texts)
    print(f"Generated {len(embeddings)} embeddings.")
    
    # Initialize Pinecone
    pinecone_api_key = "Input API KEY PINECONE"
    pc = pinecone.Pinecone(api_key=pinecone_api_key)

    index_name = "nutrition-rag-index"
    spec = ServerlessSpec(
        cloud="aws", region="us-east-1"
    )

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Create vector index if not exist
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=spec
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # Connect to index
    index = pc.Index(index_name)

    # Each entry must be a tuple of (id, vector, metadata)
    vectors = [
        {
            "id": f"vec-{i}",
            "values": embeddings[i],
            "metadata": chunk_metadata[i]
        }
        for i in range(len(embeddings))
    ]

    # Batch upsert (up to 100 vectors per call is safe)
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)

    print("Vector embeddings created successfully!")
    print(f"Stored {len(texts)} document chunks with enhanced metadata")

if __name__ == "__main__":
    main()