"""
RAG Agent for product search and information retrieval.
Uses LangChain's agent pattern with tool-based retrieval.
"""

from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from database import ProductVectorStore
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)


class ProductInfo(BaseModel):
    """Structured product information returned by the agent."""

    product_id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    description: str = Field(description="Product description")
    price: float = Field(description="Product price in dollars")
    category: str = Field(description="Product category")
    stock_status: str = Field(description="Stock availability status")


class RAGResponse(BaseModel):
    """Structured response from the RAG agent."""

    answer: str = Field(description="Natural language answer to the user's query")
    products: List[ProductInfo] = Field(
        default_factory=list, description="List of relevant products found"
    )


class RAGAgent:
    """
    RAG Agent for answering product-related queries.

    Uses LangChain's agent pattern with a retrieval tool that searches
    the vector store for relevant products.
    """

    def __init__(
        self, model_name: str = "gpt-4o-mini", temperature: float = 0, k: int = 5
    ):
        """
        Initialize the RAG Agent.

        Args:
            model_name: OpenAI model to use for generation (must support structured output)
            temperature: Sampling temperature (0 = deterministic)
            k: Number of products to retrieve from vector store
        """
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vector_store = ProductVectorStore().get()

        @tool
        def retrieve_products(query: str) -> str:
            """
            Retrieve product information to help answer customer queries.

            Searches the product catalog using semantic similarity to find relevant products
            based on the customer's question or search terms.

            Args:
                query: The customer's search query or question about products

            Returns:
                Formatted product information
            """
            logger.debug(f"Tool called: retrieve_products(query='{query}')")

            results = self.vector_store.similarity_search(query, k=self.k)

            if not results:
                serialized = "No products found matching the query."
            else:
                parts = ["Here are the relevant products:\n"]
                for i, doc in enumerate(results, 1):
                    parts.append(
                        f"{i}. {doc.metadata.get('name')}\n"
                        f"   ID: {doc.metadata.get('product_id')}\n"
                        f"   Price: ${doc.metadata.get('price'):.2f}\n"
                        f"   Category: {doc.metadata.get('category')}\n"
                        f"   Stock: {doc.metadata.get('stock_status')}\n"
                        f"   Description: {doc.page_content}\n"
                    )
                serialized = "\n".join(parts)

            logger.info(f"Retrieved {len(results)} products for query: '{query}'")

            return serialized

        model = ChatOpenAI(model=model_name, temperature=temperature)

        system_prompt = (
            "You are a helpful e-commerce product assistant. Your role is to help customers find products and answer questions about them. "
            "You have access to a tool that retrieves product information from our catalog. Use it to search for products when needed. "
            "When responding, you MUST provide a structured response with two fields: "
            "'answer' - a friendly, conversational response to the customer's query. "
            "'products' - a list of ALL relevant products from the retrieved results that match the customer's query. "
            "IMPORTANT: Include complete product details (product_id, name, description, price, category, stock_status) for each relevant product. "
            "ONLY include products that are truly relevant to the customer's query. "
            "For example, if asked about laptops, only include actual laptop computers, not laptop accessories. "
            "Highlight key features like price, category, and stock status in your answer. "
            "If a product is out of stock, mention it clearly and suggest alternatives if available. "
            "If you don't find relevant products, say so politely and ask for clarification."
        )

        self.agent = create_agent(
            model,
            tools=[retrieve_products],
            system_prompt=system_prompt,
            response_format=RAGResponse,
        )

        logger.info(
            f"RAG Agent initialized with model={model_name}, temperature={temperature}, k={k}"
        )

    def invoke(
        self, user_query: str, chat_history: Optional[List[Dict]] = None
    ) -> RAGResponse:
        """
        Answer user query using the agent.

        Args:
            user_query: User's question or search query
            chat_history: Optional list of previous messages in conversation

        Returns:
            RAGResponse with structured answer and products
        """
        logger.info(f"Processing query: '{user_query}'")

        messages = chat_history.copy() if chat_history else []
        messages.append({"role": "user", "content": user_query})

        result = self.agent.invoke({"messages": messages})

        structured_response = result.get("structured_response")

        if not structured_response:
            logger.error("Agent did not return a structured response")
            return RAGResponse(answer="I couldn't process that request.", products=[])

        logger.info(
            f"Successfully answered query with {len(structured_response.products)} products"
        )
        return structured_response
