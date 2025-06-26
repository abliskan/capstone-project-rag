import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
import hashlib
from datetime import datetime
import json
import re
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import tiktoken
from tqdm import tqdm

class CSVToPineconeProcessor:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str, embedding_model: str, chunk_size: int, chunk_overlap: int):
        """
        Initialize the CSV to Pinecone processor
        """
        # Initialize Pinecone
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass it directly.")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = index_name
        self.pinecone_environment = pinecone_environment
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize index
        self._setup_pinecone_index()
    
    def _setup_pinecone_index(self):
        """
        Setup or connect to Pinecone index
        """
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.pinecone_environment
                    )
                )
            else:
                print(f"Using existing Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to index. Current stats: {self.index.describe_index_stats()}")
            
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
            raise
    
    def load_csv_files(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Load CSV files from the data directory
        """
        csv_files = {
            'nutrition': 'External_knowledge_NutritionInfo.csv',
            'ingredient': 'External_knowledge_Ingredient.csv',
            'recipe': 'External_knowledge_Recipe.csv'
        }
        
        dataframes = {}
        
        for file_type, filename in csv_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    dataframes[file_type] = df
                    print(f"Loaded {file_type} data: {len(df)} records from {filepath}")
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
            else:
                print(f"File not found: {filepath}")
        
        return dataframes
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        """
        return len(self.tokenizer.encode(str(text)))
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap
        """
        if not text or pd.isna(text):
            return []
        
        text = str(text).strip()
        if not text:
            return []
        
        # If text is short enough, return as single chunk
        if self.count_tokens(text) <= self.chunk_size:
            return [{
                'text': text,
                'metadata': {**metadata, 'chunk_index': 0, 'total_chunks': 1}
            }]
        
        # Split into sentences for better chunking
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {**metadata, 'chunk_index': chunk_index, 'total_chunks': 0}  # Will update later
                })
                
                # Start new chunk with overlap
                overlap_text = ""
                if self.chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                    overlap_text = " ".join(overlap_words)
                
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {**metadata, 'chunk_index': chunk_index, 'total_chunks': 0}
            })
        
        # Update total_chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = total_chunks
        
        return chunks
    
    def process_nutrition_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process nutrition data into chunks"""
        chunks = []
        
        for idx, row in df.iterrows():
            # Create text representation
            text_parts = []
            if pd.notna(row.get('name')):
                text_parts.append(f"Food: {row['name']}")
            
            # Add nutritional information
            nutrition_info = []
            nutrition_fields = ['calories', 'protein_g', 'carbohydrates_total_g', 'fat_total_g', 'fiber_g', 'serving_size_g']
            
            for field in nutrition_fields:
                if field in row and pd.notna(row[field]) and row[field] != '':
                    value = row[field]
                    if field == 'serving_size_g':
                        nutrition_info.append(f"Serving size: {value}g")
                    elif field == 'calories':
                        nutrition_info.append(f"Calories: {value}")
                    else:
                        unit = field.replace('_', ' ').title()
                        nutrition_info.append(f"{unit}: {value}")
            
            if nutrition_info:
                text_parts.append("Nutrition: " + ", ".join(nutrition_info))
            
            text = ". ".join(text_parts)
            
            if text.strip():
                metadata = {
                    'source_type': 'nutrition',
                    'food_name': str(row.get('name', 'Unknown')),
                    'record_id': f"nutrition_{idx}",
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add all available fields to metadata
                for col in df.columns:
                    if pd.notna(row[col]) and row[col] != '':
                        metadata[f'nutrition_{col}'] = str(row[col])
                
                chunk_data = self.chunk_text(text, metadata)
                chunks.extend(chunk_data)
        
        return chunks
    
    def process_ingredient_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process ingredient data into chunks
        """
        chunks = []
        
        for idx, row in df.iterrows():
            text_parts = []
            
            if pd.notna(row.get('name')):
                text_parts.append(f"Ingredient: {row['name']}")
            
            if pd.notna(row.get('quantity')):
                text_parts.append(f"Quantity: {row['quantity']}")
            
            if pd.notna(row.get('category')):
                text_parts.append(f"Category: {row['category']}")
            
            text = ". ".join(text_parts)
            
            if text.strip():
                metadata = {
                    'source_type': 'ingredient',
                    'ingredient_name': str(row.get('name', 'Unknown')),
                    'record_id': f"ingredient_{idx}",
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add all available fields to metadata
                for col in df.columns:
                    if pd.notna(row[col]) and row[col] != '':
                        metadata[f'ingredient_{col}'] = str(row[col])
                
                chunk_data = self.chunk_text(text, metadata)
                chunks.extend(chunk_data)
        
        return chunks
    
    def process_recipe_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process recipe data into chunks
        """
        chunks = []
        
        for idx, row in df.iterrows():
            # Process recipe title and description
            title = str(row.get('title', 'Unknown Recipe'))
            description = str(row.get('description', ''))
            
            # Create main recipe text
            text_parts = [f"Recipe: {title}"]
            
            if description and description != 'nan':
                text_parts.append(f"Description: {description}")
            
            if pd.notna(row.get('servings')):
                text_parts.append(f"Servings: {row['servings']}")
            
            if pd.notna(row.get('cook_time_minutes')):
                text_parts.append(f"Cook time: {row['cook_time_minutes']} minutes")
            
            # Add ingredients
            if pd.notna(row.get('ingredients')):
                ingredients = str(row['ingredients'])
                text_parts.append(f"Ingredients: {ingredients}")
            
            # Add instructions
            if pd.notna(row.get('instructions')):
                instructions = str(row['instructions'])
                text_parts.append(f"Instructions: {instructions}")
            
            if pd.notna(row.get('tags')):
                tags = str(row['tags'])
                text_parts.append(f"Tags: {tags}")
            
            text = ". ".join(text_parts)
            
            if text.strip():
                metadata = {
                    'source_type': 'recipe',
                    'recipe_title': title,
                    'record_id': f"recipe_{idx}",
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add all available fields to metadata
                for col in df.columns:
                    if pd.notna(row[col]) and row[col] != '':
                        metadata[f'recipe_{col}'] = str(row[col])
                
                chunk_data = self.chunk_text(text, metadata)
                chunks.extend(chunk_data)
        
        return chunks
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for texts in batches
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings
    
    def create_vector_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Create a unique ID for a vector
        """
        content = f"{metadata.get('source_type', '')}_{metadata.get('record_id', '')}_{metadata.get('chunk_index', 0)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def upsert_to_pinecone(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upload chunks to Pinecone in batches
        """
        if not chunks:
            print("No chunks to upload")
            return
        
        print(f"Preparing to upload {len(chunks)} chunks to Pinecone...")
        
        # Extract texts for embedding generation
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = self.create_vector_id(chunk['text'], chunk['metadata'])
            vectors.append({
                'id': vector_id,
                'values': embeddings[i],
                'metadata': {
                    **chunk['metadata'],
                    'text': chunk['text'][:1000]  # Limit text size in metadata
                }
            })
        
        # Upsert in batches
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone"):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
    
    def process_all_csv_files(self, data_dir: str = "data"):
        """
        Process all CSV files and upload to Pinecone
        """
        print("Starting CSV to Pinecone processing...")
        
        # Load CSV files from 'data' folder
        dataframes = self.load_csv_files(data_dir)
        
        if not dataframes:
            print("No CSV files found to process")
            return
        
        all_chunks = []
        
        # Process each type of data
        for data_type, df in dataframes.items():
            print(f"\nProcessing {data_type} data ({len(df)} records)...")
            
            if data_type == 'nutrition':
                chunks = self.process_nutrition_data(df)
            elif data_type == 'ingredient':
                chunks = self.process_ingredient_data(df)
            elif data_type == 'recipe':
                chunks = self.process_recipe_data(df)
            else:
                print(f"Unknown data type: {data_type}")
                continue
            
            print(f"Generated {len(chunks)} chunks from {data_type} data")
            all_chunks.extend(chunks)
        
        print(f"\nTotal chunks generated: {len(all_chunks)}")
        
        if all_chunks:
            # Upload to Pinecone
            self.upsert_to_pinecone(all_chunks)
            
            # Print final stats
            stats = self.index.describe_index_stats()
            print(f"\nPinecone index stats after upload:")
            print(f"Total vectors: {stats.total_vector_count}")
            print(f"Index dimension: {stats.dimension}")
        else:
            print("No chunks generated to upload")
    
    def query_similar(self, query_text: str, top_k: int = 5, filter_dict: Dict[str, Any] = None):
        """
        Query the Pinecone index for similar content
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return results

def main():
    """
    Main function to process CSV files and upload to Pinecone
    """
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    DATA_DIR = "data"
    INDEX_NAME = "nutrition-knowledge-base"
    EMBEDDING_MODEL = "models/embedding-001"
    
    if not PINECONE_API_KEY:
        print("Please set the PINECONE_API_KEY environment variable")
        return
    
    try:
        # Initialize processor
        processor = CSVToPineconeProcessor(
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_environment=pinecone_environment,
            index_name=INDEX_NAME,
            EMBEDDING_MODEL = EMBEDDING_MODEL,
            chunk_size=512,
            chunk_overlap=50
        )
        
        # Process all CSV files
        processor.process_all_csv_files(DATA_DIR)
        
        # Example query
        print("\n=== Example Query ===")
        results = processor.query_similar("What is the protein content of chicken?", top_k=3)
        
        for i, match in enumerate(results.matches):
            print(f"\nResult {i+1} (Score: {match.score:.4f}):")
            print(f"Source: {match.metadata.get('source_type', 'Unknown')}")
            print(f"Text: {match.metadata.get('text', 'No text')[:200]}...")
        
        print("\n=== Processing Complete ===")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()