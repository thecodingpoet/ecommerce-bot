"""
RAG Agent for product search and information retrieval.
Uses LangChain's agent pattern with tool-based retrieval.
"""

import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from database import ProductVectorStore

load_dotenv()

logger = logging.getLogger(__name__)


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
    transfer_to_agent: Optional[str] = Field(
        None,
        description="Agent to transfer to: 'order' to handle purchases, or None to stay with RAG agent",
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
        def transfer_to_order_agent(reason: str) -> str:
            """
            Transfer the conversation to the Order Agent to handle purchases.

            Use this when the customer wants to:
            - Place an order or buy a product
            - Add items to cart
            - Make a purchase
            - Checkout
            - Provide payment or shipping information

            Args:
                reason: Brief explanation of why transfer is needed

            Returns:
                Confirmation message
            """
            logger.info(f"Transfer to Order Agent requested: {reason}")
            return f"TRANSFER_TO_ORDER: {reason}"

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
            "\n\n"
            "TOOLS AVAILABLE:\n"
            "1. retrieve_products - Search our product catalog\n"
            "2. transfer_to_order_agent - Transfer customer to order specialist when they want to make a purchase\n"
            "\n"
            "WHEN TO TRANSFER:\n"
            "- Customer wants to buy, purchase, or order a product\n"
            "- Customer wants to add items to cart or checkout\n"
            "- Customer wants to complete a purchase\n"
            "When transferring, set 'transfer_to_agent' to 'order' and include a friendly message in 'answer' like 'Let me connect you with our order specialist to complete your purchase.'\n"
            "\n"
            "RESPONSE FORMAT:\n"
            "You MUST provide a structured response with these fields:\n"
            "- 'answer': A friendly, conversational response to the customer's query\n"
            "- 'products': List of ALL relevant products from retrieved results (with complete details: product_id, name, description, price, category, stock_status)\n"
            "- 'transfer_to_agent': Set to 'order' when customer wants to purchase, otherwise None\n"
            "\n"
            "BEST PRACTICES:\n"
            "- ALWAYS prominently display the Product ID in your answer (e.g., 'MacBook Pro 16-inch (ID: TECH-001)')\n"
            "- Format products clearly with ID first for easy reference when ordering\n"
            "- ONLY include products truly relevant to the query (e.g., for 'laptops', exclude accessories)\n"
            "- Highlight price, category, and stock status in your answer\n"
            "- Mention if products are out of stock and suggest alternatives\n"
            "- Remind customers to use the Product ID when placing orders\n"
            "- If no relevant products found, ask for clarification politely"
        )

        self.agent = create_agent(
            model,
            tools=[retrieve_products, transfer_to_order_agent],
            system_prompt=system_prompt,
            response_format=RAGResponse,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=10,
                    exit_behavior="end",
                )
            ],
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

        try:
            result = self.agent.invoke({"messages": messages})
        except Exception as e:
            logger.error(f"Error invoking RAG agent: {e}")
            return RAGResponse(
                answer="I encountered an error searching for products. Please try again.",
                products=[],
            )

        structured_response = result.get("structured_response")

        if not structured_response:
            logger.error("Agent did not return a structured response")
            return RAGResponse(answer="I couldn't process that request.", products=[])

        logger.info(
            f"Successfully answered query with {len(structured_response.products)} products"
        )
        return structured_response
