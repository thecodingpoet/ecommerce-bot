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


from database import ProductCatalog, ProductVectorStore
from schema import RAGResponse

load_dotenv()

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent for answering product-related queries.

    Uses LangChain's agent pattern with a retrieval tool that searches
    the vector store for relevant products.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        k: int = 5,
        timeout: int = 30,
    ):
        """
        Initialize the RAG Agent.

        Args:
            model_name: OpenAI model to use for generation (must support structured output)
            temperature: Sampling temperature (0 = deterministic)
            k: Number of products to retrieve from vector store
            timeout: Request timeout in seconds (default: 30)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.timeout = timeout
        self.vector_store = ProductVectorStore().get()
        self.product_catalog = ProductCatalog()

        def format_stock_status(stock_status: str) -> str:
            """Format stock status for display."""
            status_map = {
                "in_stock": "In Stock",
                "low_stock": "Low Stock",
                "out_of_stock": "Out of Stock",
            }
            return status_map.get(stock_status, stock_status)

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

            - If the query matches a product ID or exact name, return product details.
            - Otherwise, perform semantic search and return a list of relevant products.

            Args:
                query: The customer's search query, product name, or product ID

            Returns:
                Formatted product information
            """
            # 1. Exact lookup (ID or exact name)
            product = self.product_catalog.get_product_by_id_or_name(query)

            if product:
                name = product["name"]
                pid = product["product_id"]
                price = product["price"]
                stock = format_stock_status(product["stock_status"])
                description = product["description"]

                logger.info(f"Exact product match found: {pid}")

                return (
                    f"**{name}** (ID: {pid})\n"
                    f"Price: ${price:.2f} | Stock: {stock}\n"
                    f"Description: {description}\n"
                    f"How many units would you like to order?"
                )

            results = self.vector_store.similarity_search(query, k=self.k)

            if not results:
                return "No products found matching your search."

            lines = []
            for i, doc in enumerate(results, 1):
                meta = doc.metadata
                lines.append(
                    f"{i}. **{meta['name']}** (ID: {meta['product_id']}) "
                    f"- ${meta['price']:.2f} | {format_stock_status(meta['stock_status'])}"
                )

            logger.info(
                f"Semantic search returned {len(results)} results for '{query}'"
            )
            return "\n".join(lines)

        model = ChatOpenAI(model=model_name, temperature=temperature, timeout=timeout)

        system_prompt = (
            "You are a helpful e-commerce product assistant. Your role is to help customers find products and answer questions about them. "
            "\n\n"
            "TOOLS AVAILABLE:\n"
            "1. retrieve_products - Search our product catalog\n"
            "   Use this tool to search for products by name, category, features, or product ID\n"
            "   The tool automatically formats results appropriately:\n"
            "   - For browsing/searching: Returns numbered list format\n"
            "   - For specific product queries: Returns detailed product information\n"
            "\n"
            "2. transfer_to_order_agent - Transfer customer to order specialist when they want to make a purchase\n"
            "\n"
            "WHEN TO TRANSFER:\n"
            "- Customer wants to buy, purchase, or order a product\n"
            "- Customer wants to add items to cart or checkout\n"
            "- Customer wants to complete a purchase\n"
            "When transferring, set 'transfer_to_agent' to 'order' and include a friendly message like 'Let me connect you with our order specialist to complete your purchase.'\n"
            "\n"
            "RESPONSE FORMAT:\n"
            "You MUST provide a structured response with these fields:\n"
            "- 'message': A friendly, conversational response to the customer's query\n"
            "- 'products': List of ALL relevant products from retrieved results (with complete details: product_id, name, description, price, category, stock_status)\n"
            "- 'transfer_to_agent': Set to 'order' when customer wants to purchase, otherwise None\n"
            "\n"
            "BEST PRACTICES:\n"
            "- ALWAYS prominently display the Product ID in your message (e.g., 'MacBook Pro 16-inch (ID: TECH-001)')\n"
            "- Format products clearly with ID first for easy reference when ordering\n"
            "- ONLY include products truly relevant to the query (e.g., for 'laptops', exclude accessories)\n"
            "- Highlight price, category, and stock status in your message\n"
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
            f"RAG Agent initialized with model={model_name}, temperature={temperature}, k={k}, timeout={timeout}s"
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
            is_timeout = (
                isinstance(e, TimeoutError)
                or "timeout" in type(e).__name__.lower()
                or "timeout" in str(e).lower()
            )

            if is_timeout:
                logger.warning(f"Timeout exceeded ({self.timeout}s) in RAG agent: {e}")
                return RAGResponse(
                    message=(
                        "Searching our catalog is taking longer than usual. "
                        "This might be due to high traffic. Please try your search again, "
                        "or try being more specific about what you're looking for."
                    ),
                    products=[],
                )
            else:
                logger.error(f"Error invoking RAG agent: {e}", exc_info=True)
                return RAGResponse(
                    message="I encountered an error searching for products. Please try again.",
                    products=[],
                )

        structured_response = result.get("structured_response")

        if not structured_response:
            logger.debug(
                "RAG Agent did not return a structured response - LLM may have had trouble determining intent"
            )
            return RAGResponse(
                message=(
                    "I'm having trouble understanding your search. Could you try rephrasing?\n"
                    "• Try being more specific about what you're looking for\n"
                    "• You can search by product name, category, or features\n"
                    "• For example: 'show me laptops' or 'wireless headphones under $100'"
                ),
                products=[],
            )

        logger.info(
            f"Successfully answered query with {len(structured_response.products)} products"
        )
        return structured_response
