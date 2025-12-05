"""
Script to initialize ChromaDB vector store with product embeddings.

This is a convenience script that wraps the ProductVectorStore class
from the database module. Use this to quickly initialize or reset
the vector store from the command line.

Usage:
    cd src && python initialize_vector_store.py
"""

from dotenv import load_dotenv

from database import ProductVectorStore
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("Initializing vector store...")
    store = ProductVectorStore()
    vector_store = store.initialize()
    logger.info("Vector store ready!")
