import os
import re
import requests
# import uuid
from dotenv import load_dotenv
from pinecone import Pinecone
from time import sleep, time
from langfuse.langchain import CallbackHandler
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Langfuse Setup with Safe Fallback
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    from langfuse.decorators import observe
    LANGFUSE_ENABLED = True
    print("[SourchefBot] Langfuse loaded successfully!")
except ImportError:
    LANGFUSE_ENABLED = False
    print("[SourchefBot] Langfuse not installed. Using mock tracing...")
    
    # Mock decorator for when Langfuse is not available
    def observe(name=None):
        def decorator(func):
            return func
        return decorator

# Load environment variables from .env file
load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
google_api_key=os.getenv("GOOGLE_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST")


# Initialize Langfuse client
langfuse_client = None
if LANGFUSE_ENABLED:
    try:
        langfuse_client = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host=langfuse_host
        )
        
        # Verify connection
        if langfuse_client.auth_check():
            print("Langfuse client is authenticated and ready!")
        else:
            print("Langfuse authentication failed. Check your credentials.")
            langfuse_client = None
    except Exception as e:
        print(f"Failed to initialize Langfuse: {e}")
        langfuse_client = None

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Create vector index if not exist
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # depends on your embedding size (Gemini is 768)
        metric="cosine",  # or "euclidean", "dotproduct"
        pod_type="p1"  # example: small pod
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to index
index = pc.Index(index_name)

# LangChain LLM Setup
callbacks = [CallbackHandler()]

# Embeddings model and LLM model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Gemini embedding model
    google_api_key=google_api_key,
    callbacks=callbacks,
    verbose=True
)

llm = GoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0,
                             max_tokens=None,
                             timeout=None)

# Load Pinecone index
# vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)
retriever = vectorstore.as_retriever(search_type="similarity", k=4)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are SourchefBot, a helpful and personalized AI nutrition and chef assistant.

Chat History:
{chat_history}

Relevant Context:
{context}

User question:
{question}

SourchefBot response:
"""
)

# Build LLM chain
llm_chain = LLMChain(llm=llm, 
    prompt=prompt_template,
    callbacks=callbacks
)

stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

# RAG + Memory Chain + Langfuse tracing
chat_rag = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    callbacks=callbacks,
    verbose=True
    # observability=True  # enables Langfuse
)

# Estimate nutrition with Langfuse tracing
@observe(name="estimate_nutrition")
def estimate_nutrition(ingredients_text):
    if not ingredients_text:
        return {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
    
    words = ingredients_text.lower().split()
    calories = 400 + 10 * words.count("chicken") - 5 * words.count("salad")
    protein = 20 + 3 * words.count("egg")
    fat = 15 + 2 * words.count("cheese")
    carbs = 50 - 3 * words.count("meat")
    nutrition_data = {
        "calories": max(0, calories), 
        "protein": max(0, protein), 
        "fat": max(0, fat), 
        "carbs": max(0, carbs)
    }
    
    # Log to Langfuse if available
    if LANGFUSE_ENABLED and langfuse_client:
        try:
            langfuse_client.score(
                name="nutrition_estimation",
                value=calories,
                data_type="NUMERIC"
            )
        except Exception as e:
            print(f"Failed to log nutrition score to Langfuse: {e}")
    
    return nutrition_data

# Get cooking videos using YouTube API with Langfuse tracing
@observe(name="fetch_youtube_videos")
def fetch_youtube_videos(query):
    if not youtube_api_key:
        print("YouTube API key not found")
        return []
    
    api_key = os.getenv("YOUTUBE_API_KEY")
    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"{query} recipe",
        "type": "video",
        "maxResults": 3,
        "key": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        videos = []
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            title = snippet.get("title", "Untitled Video")
            link = f"https://www.youtube.com/watch?v={video_id}"
            embed = f"https://www.youtube.com/embed/{video_id}"

            videos.append({
                "title": title,
                "link": link,
                "embed": embed
            })
        
        # Log success to Langfuse
        if LANGFUSE_ENABLED and langfuse_client:
            try:
                langfuse_client.score(
                    name="youtube_api_success",
                    value=len(videos),
                    data_type="NUMERIC"
                )
            except Exception as e:
                print(f"Failed to log YouTube API score to Langfuse: {e}")
        
        return videos
    except Exception as e:
        print(f"[YouTube API ERROR] {e}")
        return []

# Main RAG Handler with comprehensive Langfuse tracing
@observe(name="get_memory_rag_answer")
def get_memory_rag_answer(query: str, user_id: str = "anonymous"):
    try:
        # Create a trace for this request if Langfuse is available
        trace = None
        if LANGFUSE_ENABLED and langfuse_client:
            trace = langfuse_client.trace(
                name="sourchef_bot_query",
                user_id=user_id,
                input={"query": query},
                metadata={
                    "model": "gemini-2.0-flash",
                    "retriever_k": 4,
                    "vector_store": "pinecone"
                }
            )
        
        # Get response from RAG chain
        if LANGFUSE_ENABLED and trace:
            with trace.span(name="rag_chain_execution", input={"query": query}) as span:
                response = chat_rag.run(query)
                span.end(output={"response": response})
        else:
            response = chat_rag.run(query)
    
        title = re.search(r"^([^\n]+)", response.strip()).group(1).strip()
        match = re.search(r"(?i)ingredients:?(.+?)instructions:", response, re.DOTALL)
        
        # Extract ingredients for macro calc
        ingredients = match.group(1).strip() if match else ""
        nutrition = estimate_nutrition(response)
        videos = fetch_youtube_videos(title)    
        
        return {
            "title": title,
            "answer": response,
            "nutrition": nutrition,
            "videos": videos,
            "ingredients": ingredients,
            "user_id": user_id
        }
    except Exception as e:
        error_msg = f"Error in get_memory_rag_answer: {str(e)}"
        print(error_msg)
