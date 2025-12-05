"""Initialize ChromaDB vector store with product embeddings using LangChain."""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _get_embeddings() -> OpenAIEmbeddings:
    """Get OpenAI embeddings instance.

    Returns:
        Configured OpenAIEmbeddings instance
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")


def initialize_vector_store(
    products_path: str = "data/products.json",
    persist_directory: str = "data/chroma_db",
    collection_name: str = "products",
) -> Chroma:
    """Initialize or reset vector store with product embeddings.

    Args:
        products_path: Path to products.json file
        persist_directory: Path to ChromaDB persistent storage
        collection_name: Name of the collection to create

    Returns:
        Chroma vector store with embedded products

    Raises:
        FileNotFoundError: If products.json doesn't exist
    """
    products_file = Path(products_path)
    if not products_file.exists():
        raise FileNotFoundError(f"Products file not found: {products_path}")

    with open(products_file) as f:
        products = json.load(f)

    logger.info(f"Loaded {len(products)} products from {products_path}")

    embeddings = _get_embeddings()

    persist_path = Path(persist_directory)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        logger.info(f"Deleted existing vector store at {persist_directory}")

    documents = []
    for product in products:
        page_content = f"{product['name']}. {product['description']}"

        metadata = {
            "product_id": product["product_id"],
            "name": product["name"],
            "price": product["price"],
            "category": product["category"],
            "stock_status": product["stock_status"],
        }

        doc = Document(
            page_content=page_content, metadata=metadata, id=product["product_id"]
        )
        documents.append(doc)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    logger.info(f"Added {len(documents)} products to vector store")
    logger.info(f"Collection saved to {persist_directory}")

    return vector_store


def get_vector_store(
    persist_directory: str = "data/chroma_db",
    collection_name: str = "products",
) -> Chroma:
    """Get existing vector store collection.

    Args:
        persist_directory: Path to ChromaDB persistent storage
        collection_name: Name of the collection

    Returns:
        Chroma vector store

    Raises:
        ValueError: If collection doesn't exist
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise ValueError(
            f"Vector store not found at {persist_directory}. Run initialize_vector_store() first."
        )

    embeddings = _get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    logger.info("Initializing vector store...")
    vector_store = initialize_vector_store()
    logger.info("Vector store ready!")
