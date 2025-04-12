"""
Property Search Module

This module handles the AI-powered property search functionality. It includes:
- Integration with Google's Gemini AI for query understanding
- Vector search using Pinecone and HuggingFace embeddings
- BM25 text matching for hybrid search capabilities
- Prompt generation for initial and refined searches

The module extracts structured metadata from natural language queries and
uses this to find relevant properties that match the user's criteria.
"""

import json
import nltk
import os
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
import streamlit as st
from streamlit.logger import get_logger
import logging

# Configure logging
LOGGER = get_logger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.debug(f'start of streamlit_test')

model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",cache_folder='./PropertySearch')

# Set up Gemini AI
api_key_gemini = st.secrets["GEMINI_API"]
api_key_pinecone = st.secrets["PINECONE_API"]
genai.configure(api_key=api_key_gemini)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
LOGGER.debug(gemini_model)

# Ensure NLTK data is available for BM25
try:
    nltk.data.find('C:/nltk_data/corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up Pinecone vector database
index_name = "airbnb-property-search"
pc = Pinecone(api_key=api_key_pinecone)
index = pc.Index(index_name)

# Initialize BM25 encoder for text matching
bm25 = BM25Encoder()

# Create hybrid retriever combining vector and sparse search
retriever = PineconeHybridSearchRetriever(embeddings=model, sparse_encoder=bm25, index=index, top_k=10)
LOGGER.debug(retriever)


def call_gemini_api(prompt):
    """
    Call Google's Gemini AI to process the provided prompt.
    
    Args:
        prompt (str): The prompt text to send to Gemini API
        
    Returns:
        dict: Parsed JSON response from Gemini API
    """
    response = gemini_model.generate_content(prompt, generation_config=genai.GenerationConfig(
        response_mime_type="application/json"))
    response = json.loads(response.text)
    LOGGER.debug(response)
    return response


def generate_initial_prompt(user_input: str):
    """
    Generate the initial prompt for Gemini AI to extract metadata from user input.
    
    Args:
        user_input (str): The user's property search description
        
    Returns:
        str: Formatted prompt for Gemini API
    """
    prompt = """
    You are a professional real estate copywriter and metadata extractor.
    Your task is to generate engaging descriptions for properties while also providing structured metadata for each property.

    ### **Task 1: Property Description**
    Create a detailed, inviting description for each property based on the provided information. Highlight unique features, amenities, and location benefits in an appealing tone. Ensure to include details such as city, neighborhood, number of guests the property can accommodate, number of bedrooms and bathrooms, and the price (if not `-1`).

    ### **Task 2: Metadata Extraction**
    Extract structured metadata for each property in the following format:
    ```json
    {
      "City_name": "[City Name, or 'Unknown' if not specified]",
      "neighborhood_name": "[Neighborhood Name, or 'Unknown' if not specified]",
      "imp_amenities": ["[List of amenities, such as Wi-Fi, Pool, Kitchen, etc., inferred or leave empty if not provided]"],
      "accommodates": "[Number of guests the property can accommodate, or 'Unknown' if not specified]",
      "bathrooms": "[Number of bathrooms, or 'Unknown' if not specified]",
      "bedrooms": "[Number of bedrooms, or 'Unknown' if not specified]",
      "price": "[Price range in the format '<=100', '<=200', '<=300', etc., or 'Unknown' if not specified]"
    }
    ```
    Example output:
    ```json
    {"Properties":[
    {
        "Property ID": "ID of the property",
        "description": "Located in the heart of San Diego's vibrant Gaslamp Quarter, this chic apartment is perfect for travelers looking for a modern retreat. With 2 bedrooms, 2 bathrooms, and space for 4 guests, the property offers amenities like Wi-Fi, a fully equipped kitchen, and a rooftop pool. At $150 per night, enjoy easy access to local attractions such as the San Diego Zoo and Seaport Village.",
        "metadata": {
          "City_name": "San Diego",
          "neighborhood_name": "Gaslamp Quarter",
          "imp_amenities": ["Wi-Fi", "Kitchen", "Rooftop Pool", "Air Conditioning", "TV", "Parking", "Washer", "Dryer", "Gym", "Balcony"],
          "accommodates": 4,
          "bathrooms": "2",
          "bedrooms": 2,
          "price": "<=200"
        }

    }, ...]

    ```

    ### **General Rules**
    1. If any data is missing or cannot be inferred, mark it as `'Unknown'`.
    2. For the `price`, categorize into ranges like `<100`, `<200`, `<300`, etc., based on the given data. If no price is provided, use `'Unknown'`.
    
    """
    return f"{prompt} Here is the property description provided by the user {user_input}"


def generate_refined_prompt(input_data: dict):
    """
    Generate a prompt for refining a search based on previous search context and new input.
    
    Args:
        input_data (dict): Dictionary containing previous query, metadata, and current query
        
    Returns:
        str: Formatted prompt for Gemini API
    """
    prompt = f"""
    You are an intelligent assistant that updates metadata and generates a refined query based on structured input. 
    
    Your tasks are:

    1. Extract new details from the `current_query` and update the `previous_metadata`. If a detail is not mentioned in 
    the `current_query`, retain the value from the `previous_metadata`.
    2. Generate a refined query by combining the `prev_query` and the `current_query` while ensuring it aligns with the 
    updated metadata..

### **Task 1: Property Description**
Create a detailed, inviting description for each property based on the provided information. 
Highlight unique features, amenities, and location benefits in an appealing tone. Ensure to include details such as city,
neighborhood, number of guests the property can accommodate, number of bedrooms and bathrooms, and the price
Generate a refined query by combining the `prev_query` and the `current_query` 
while ensuring it aligns with the updated metadata.

    ### Input:
    {input_data}

    ### Output:
    Respond in this exact JSON format:
    {{
      "updated_metadata": {{
        "City_name": "[Updated City Name or 'Unknown']",
        "neighborhood_name": "[Updated Neighborhood Name or 'Unknown']",
        "imp_amenities": ["[Updated list of amenities]"],
        "accommodates": "[Updated number of guests or 'Unknown']",
        "bathrooms": "[Updated number of bathrooms or 'Unknown']",
        "bedrooms": "[Updated number of bedrooms or 'Unknown']",
        "price": "[Updated price(<=100,<=200,<=300) or 'Unknown']"
      }},
      "refined_query": "[Refined query in natural language based on the updated metadata]"
    }}
    """
    return prompt


def search_similar_properties(user_input, metadata):
    """
    Search for properties matching the user's criteria using hybrid search.
    
    Args:
        user_input (str): User's search query or description
        metadata (dict): Structured metadata extracted from the query
        
    Returns:
        list: List of document objects for matching properties, or None if no matches
    """
    # Build filters based on metadata
    filters = {}
    for key, value in metadata.items():
        if value != 'Unknown':
            if key == 'City_name':
                filters['City_name'] = {"$eq": value}
            elif key == 'accommodates':
                filters['accommodates'] = {"$gte": value}
            elif key == 'price':
                filters['price'] = {"$eq": value}
            elif key == 'neighborhood_name':
                filters['neighborhood_name'] = {"$eq": value}

    if len(filters.keys()) == 0:
        filters = None
        
    # Perform the search
    try:
        LOGGER.debug(user_input)
        docs = retriever.get_relevant_documents(query=user_input, metadata=filters)
    except Exception as e:
        docs = None

    return docs
