"""Product data management including catalog and vector store."""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ProductCatalog:
    """Manages access to product catalog data."""

    def __init__(self, products_path: str = "data/products.json"):
        """
        Initialize ProductCatalog.

        Args:
            products_path: Path to products JSON file
        """
        self.products_path = products_path
        self._products: List[Dict] = []
        self._load_products()

    def _load_products(self):
        """Load products from JSON file."""
        products_file = Path(self.products_path)
        if not products_file.exists():
            raise FileNotFoundError(f"Products file not found: {self.products_path}")

        with open(products_file) as f:
            self._products = json.load(f)

        logger.info(f"Loaded {len(self._products)} products from {self.products_path}")

    def get_product(self, product_id: str) -> Optional[Dict]:
        """
        Get product by ID.

        Args:
            product_id: Product identifier

        Returns:
            Product dict or None if not found
        """
        for product in self._products:
            if product["product_id"] == product_id:
                return product
        return None

    def get_product_by_id_or_name(self, query: str) -> Optional[Dict]:
        """
        Get product by exact ID or exact name (case-insensitive).

        First attempts to match by exact product ID using get_product().
        If no match is found, attempts to match by exact product name (case-insensitive).

        Args:
            query: Product identifier or exact product name

        Returns:
            Product dict or None if not found
        """
        product = self.get_product(query)
        if product:
            return product

        query_lower = query.lower()
        for product in self._products:
            if product["name"].lower() == query_lower:
                return product

        return None

    def is_available(self, product_id: str) -> bool:
        """
        Check if product is available for order.

        Args:
            product_id: Product identifier

        Returns:
            True if product is in stock or low stock
        """
        product = self.get_product(product_id)
        if not product:
            return False
        return product["stock_status"] in ["in_stock", "low_stock"]

    def get_all_products(self) -> List[Dict]:
        """Get all products."""
        return self._products.copy()


class ProductVectorStore:
    """Manages ChromaDB vector store for product embeddings."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "products",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize ProductVectorStore.

        Args:
            persist_directory: Path to ChromaDB persistent storage
            collection_name: Name of the collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def initialize(self, products_path: str = "data/products.json") -> Chroma:
        """
        Initialize or reset vector store with product embeddings.

        Args:
            products_path: Path to products.json file

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

        persist_path = Path(self.persist_directory)
        if persist_path.exists():
            shutil.rmtree(persist_path)
            logger.info(f"Deleted existing vector store at {self.persist_directory}")

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
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )

        logger.info(f"Added {len(documents)} products to vector store")
        logger.info(f"Collection saved to {self.persist_directory}")

        return vector_store

    def get(self) -> Chroma:
        """
        Get existing vector store collection.

        Returns:
            Chroma vector store

        Raises:
            ValueError: If collection doesn't exist
        """
        persist_path = Path(self.persist_directory)
        if not persist_path.exists():
            raise ValueError(
                f"Vector store not found at {self.persist_directory}. "
                f"Run initialize() first."
            )

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        return vector_store
