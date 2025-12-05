"""
RAG Agent for product search and information retrieval.
Uses LangChain's agent pattern with tool-based retrieval.
"""

from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from database import ProductVectorStore
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)


class RAGAgent:
    """
    RAG Agent for answering product-related queries.

    Uses LangChain's agent pattern with a retrieval tool that searches
    the vector store for relevant products.
    """

    def __init__(
        self, model_name: str = "gpt-3.5-turbo", temperature: float = 0, k: int = 5
    ):
        """
        Initialize the RAG Agent.

        Args:
            model_name: OpenAI model to use for generation
            temperature: Sampling temperature (0 = deterministic)
            k: Number of products to retrieve from vector store
        """
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vector_store = ProductVectorStore().get()

        # Create the retrieval tool as instance method
        @tool(response_format="content_and_artifact")
        def retrieve_products(query: str) -> tuple[str, List[Dict[str, Any]]]:
            """
            Retrieve product information to help answer customer queries.

            Searches the product catalog using semantic similarity to find relevant products
            based on the customer's question or search terms.

            Args:
                query: The customer's search query or question about products

            Returns:
                Formatted product information and raw product data
            """
            logger.debug(f"Tool called: retrieve_products(query='{query}')")

            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=self.k)

            # Convert to list of dicts with metadata
            products = []
            for doc in results:
                product = {
                    "product_id": doc.metadata.get("product_id"),
                    "name": doc.metadata.get("name"),
                    "description": doc.page_content,
                    "price": doc.metadata.get("price"),
                    "category": doc.metadata.get("category"),
                    "stock_status": doc.metadata.get("stock_status"),
                }
                products.append(product)

            # Format for LLM context
            if not products:
                serialized = "No products found matching the query."
            else:
                parts = ["Here are the relevant products:\n"]
                for i, product in enumerate(products, 1):
                    parts.append(
                        f"{i}. {product['name']}\n"
                        f"   ID: {product['product_id']}\n"
                        f"   Price: ${product['price']:.2f}\n"
                        f"   Category: {product['category']}\n"
                        f"   Stock: {product['stock_status']}\n"
                        f"   Description: {product['description']}\n"
                    )
                serialized = "\n".join(parts)

            logger.info(f"Retrieved {len(products)} products for query: '{query}'")

            return serialized, products

        model = ChatOpenAI(model=model_name, temperature=temperature)

        system_prompt = (
            "You are a helpful e-commerce product assistant. Your role is to help customers find products and answer questions about them. "
            "You have access to a tool that retrieves product information from our catalog. Use it to search for products when needed. "
            "When responding: "
            "Be friendly and conversational. "
            "Provide accurate information based on the retrieved product data. "
            "Highlight key features like price, category, and stock status. "
            "If a product is out of stock, mention it clearly and suggest alternatives if available. "
            "If you don't find relevant products, say so politely and ask for clarification. "
            "Format your responses in a natural, helpful way. Don't just list products - engage with the customer's specific question."
        )

        # Create agent with retrieval tool
        self.agent = create_agent(
            model, tools=[retrieve_products], system_prompt=system_prompt
        )

        logger.info(
            f"RAG Agent initialized with model={model_name}, temperature={temperature}, k={k}"
        )

    def invoke(self, user_query: str) -> Dict[str, Any]:
        """
        Answer user query using the agent.

        Args:
            user_query: User's question or search query

        Returns:
            Dictionary with:
                - answer: Natural language response
                - products: List of retrieved products (if any)
                - messages: Full conversation history
        """
        logger.info(f"Processing query: '{user_query}'")

        # Invoke agent
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]}
        )

        messages = result["messages"]
        answer = messages[-1].content if messages else ""

        products = []
        for msg in messages:
            if hasattr(msg, "artifact") and msg.artifact:
                if isinstance(msg.artifact, list):
                    products.extend(msg.artifact)

        logger.info(
            f"Successfully answered query with {len(products)} products retrieved"
        )

        return {"answer": answer, "products": products, "messages": messages}
