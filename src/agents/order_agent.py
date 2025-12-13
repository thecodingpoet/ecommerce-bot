"""
Order Agent for handling purchase requests and order creation.
Uses LangChain's agent pattern with tools for order operations.
"""

import json
import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from database import OrderDatabase, ProductCatalog
from schema import OrderResponse

load_dotenv()

logger = logging.getLogger(__name__)


class OrderAgent:
    """
    Order Agent for processing customer orders.

    Uses LangChain's agent pattern with tools for product validation
    and order creation.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        timeout: int = 30,
    ):
        """
        Initialize the Order Agent.

        Args:
            model_name: OpenAI model to use (must support structured output)
            temperature: Sampling temperature (0 = deterministic)
            timeout: Request timeout in seconds (default: 30)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.catalog = ProductCatalog()
        self.order_db = OrderDatabase()

        @tool
        def transfer_to_rag_agent(reason: str) -> str:
            """
            Transfer the conversation back to the Product Search Agent.

            Use this when the customer wants to:
            - Search for products
            - Browse the catalog
            - Get product information or specifications
            - Compare products
            - Ask about product availability

            Args:
                reason: Brief explanation of why transfer is needed

            Returns:
                Confirmation message
            """
            logger.info(f"Transfer to RAG Agent requested: {reason}")
            return f"TRANSFER_TO_RAG: {reason}"

        @tool
        def validate_product(product_id: str, quantity: int = 1) -> str:
            """
            Validate product availability and get pricing information.

            Use this when customer mentions a product they want to order.
            Check if product exists, is available, and get current price.

            Args:
                product_id: The product ID to validate
                quantity: Quantity customer wants (default: 1)

            Returns:
                Validation result with product details and availability
            """
            logger.debug(
                f"Tool called: validate_product(product_id='{product_id}', quantity={quantity})"
            )

            product = self.catalog.get_product(product_id)

            if not product:
                return f"Product {product_id} not found. Please check the product ID or search for the product first."

            is_available = self.catalog.is_available(product_id)
            stock_status = product["stock_status"]

            if not is_available:
                return (
                    f"Sorry, {product['name']} (ID: {product_id}) is currently out of stock. "
                    f"Would you like me to suggest similar products?"
                )

            total = product["price"] * quantity

            result = (
                f"✓ Product validated:\n"
                f"- Name: {product['name']}\n"
                f"- Product ID: {product_id}\n"
                f"- Price: ${product['price']:.2f} each\n"
                f"- Quantity: {quantity}\n"
                f"- Subtotal: ${total:.2f}\n"
                f"- Stock: {stock_status.replace('_', ' ').title()}\n"
            )

            if stock_status == "low_stock":
                result += "\nNote: This item has limited stock available."

            return result

        @tool
        def create_order(
            items: str,
            customer_name: str,
            email: str,
            shipping_address: str,
        ) -> str:
            """
            Create a new order in the system.

            Use this ONLY after you have:
            1. Validated ALL products
            2. Collected ALL customer information (name, email, address)
            3. Confirmed the order with the customer

            Args:
                items: JSON string of items list. Each item must have: product_id, quantity
                       Example: '[{"product_id": "TECH-007", "quantity": 2}, {"product_id": "HOME-004", "quantity": 1}]'
                customer_name: Customer's full name
                email: Customer's email address
                shipping_address: Complete shipping address

            Returns:
                Order confirmation with order ID and details
            """

            logger.debug(f"Tool called: create_order with items: {items}")

            try:
                items_list = json.loads(items)
            except json.JSONDecodeError:
                return f"Error: Invalid items format. Must be a JSON array."

            if not items_list or not isinstance(items_list, list):
                return f"Error: Items must be a non-empty list."

            order_items = []
            total = 0
            items_summary = []

            for item in items_list:
                product_id = item.get("product_id")
                quantity = item.get("quantity", 1)

                product = self.catalog.get_product(product_id)
                if not product:
                    return f"Error: Product {product_id} not found."

                if not self.catalog.is_available(product_id):
                    return f"Error: {product['name']} is out of stock."

                subtotal = product["price"] * quantity
                total += subtotal

                order_items.append(
                    {
                        "product_id": product_id,
                        "product_name": product["name"],
                        "quantity": quantity,
                        "unit_price": product["price"],
                    }
                )

                items_summary.append(
                    f"- {quantity}x {product['name']} @ ${product['price']:.2f} each = ${subtotal:.2f}"
                )

            order = self.order_db.create_order(
                customer_name=customer_name,
                customer_email=email,
                items=order_items,
            )

            logger.info(
                f"Order created: {order.order_id} with {len(order_items)} items"
            )

            items_text = "\n".join(items_summary)

            return (
                f"✅ Order placed successfully!\n\n"
                f"Order ID: {order.order_id}\n"
                f"Customer: {customer_name}\n"
                f"Email: {email}\n\n"
                f"Items:\n{items_text}\n\n"
                f"Total: ${total:.2f}\n\n"
                f"Shipping to: {shipping_address}\n\n"
                f"Your order has been confirmed! Order ID: {order.order_id}. Total: ${total:.2f}. Thank you!"
            )

        model = ChatOpenAI(model=model_name, temperature=temperature, timeout=timeout)

        system_prompt = (
            "You are a helpful order assistant for an e-commerce store. Your role is to help customers place orders. "
            "You have access to the full conversation history, so ALWAYS use it to understand context and follow-up responses. "
            "\n\n"
            "USING CHAT HISTORY:\n"
            "- The conversation history is available in the messages you receive\n"
            "- To find product_id when customer mentions product by name:\n"
            "  1. Look through previous assistant messages for product listings\n"
            "  2. Search for patterns like 'Product ID: TECH-001' or '(ID: TECH-001)' or '**Product ID:** TECH-001'\n"
            "  3. Match the product name mentioned by customer to the name in previous listings\n"
            "  4. Extract the product_id from that listing\n"
            "- Example: If customer says 'I want the macbook' and earlier you see 'MacBook Pro 16-inch (ID: TECH-001)', use TECH-001\n"
            "- If product_id cannot be found in chat history, ask customer to search for the product first\n"
            "\n"
            "TOOLS AVAILABLE:\n"
            "1. validate_product - Check product availability and pricing\n"
            "2. create_order - Create the final order\n"
            "3. transfer_to_rag_agent - Transfer customer back to product search\n"
            "\n"
            "WHEN TO TRANSFER:\n"
            "- Customer wants to search for products\n"
            "- Customer wants to browse catalog or get product info\n"
            "- Customer wants product recommendations or comparisons\n"
            "When transferring: Use transfer_to_rag_agent tool, set 'transfer_to_agent' field to 'rag', and include a friendly message.\n"
            "\n"
            "ORDER PROCESS:\n"
            "1. VALIDATE PRODUCTS: Use validate_product tool for EACH product_id and quantity\n"
            "   - Look in chat history to find product_id when customer mentions product by name\n"
            "   - If product_id cannot be found in chat history, ask customer to search for the product first or provide the Product ID\n"
            "   - Extract quantity from customer's message:\n"
            "     * Numbers: 'I want 2 macbook' → quantity = 2\n"
            "     * Number words: 'I want three laptops' → quantity = 3\n"
            "     * Articles: 'I want to buy a macbook' → quantity = 1\n"
            "     * No quantity mentioned: 'I want macbook' → quantity = 1 (infer from context)\n"
            "   - If quantity is clear (explicit number or article), validate immediately\n"
            "   - If quantity is truly ambiguous, ask 'How many would you like to order?'\n"
            "2. ASK TO ADD MORE: After each validation, ask 'Would you like to add anything else to your order?'\n"
            "3. COLLECT INFO: Once done shopping, collect - Name, Email, Full shipping address\n"
            "4. CONFIRM: Show summary with ALL items (with quantities) and total, ask for final confirmation\n"
            "5. CREATE: Use create_order with ALL items as JSON array\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- Extract quantity from customer's message FIRST before asking questions\n"
            "- Validate ALL products using validate_product tool before collecting customer info\n"
            "- Use ONLY product information from validate_product tool results - never invent prices, names, or details\n"
            "- If validate_product returns an error (product not found, out of stock), use that exact information\n"
            "- After each validation, ask about adding more items\n"
            "- Support multiple items in a single order\n"
            '- Items format: \'[{"product_id": "TECH-007", "quantity": 2}]\'\n'
            "- Never create order without confirmation\n"
            "- Handle out-of-stock by offering alternatives or transferring to search\n"
            "- Ask for one detail at a time if not all provided\n"
            "\n"
            "RESPONSE FORMAT:\n"
            "- 'status': MUST be one of: 'collecting_info', 'confirming', 'completed', 'failed'\n"
            "  * 'collecting_info': Validating products, collecting customer details (name/email/address), or asking about adding more items\n"
            "  * 'confirming': Showing order summary and waiting for final confirmation\n"
            "  * 'completed': Order successfully created with order ID\n"
            "  * 'failed': ONLY when cannot fulfill order (e.g., all products out of stock, invalid product)\n"
            "- 'transfer_to_agent': Set to 'rag' when customer wants to search/browse products, otherwise None\n"
            "- 'message': Your friendly response to the customer\n"
            "- NEVER use status='failed' just to ask for information - use 'collecting_info' instead!"
        )

        self.agent = create_agent(
            model,
            tools=[transfer_to_rag_agent, validate_product, create_order],
            system_prompt=system_prompt,
            response_format=OrderResponse,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=10,
                    exit_behavior="end",
                ),
                ToolCallLimitMiddleware(
                    tool_name="validate_product",
                    run_limit=5,
                ),
                ToolCallLimitMiddleware(
                    tool_name="create_order",
                    run_limit=1,
                ),
            ],
        )

        logger.info(
            f"Order Agent initialized with model={model_name}, temperature={temperature}, timeout={timeout}s"
        )

    def invoke(
        self, user_query: str, chat_history: Optional[List[Dict]] = None
    ) -> OrderResponse:
        """
        Process customer order request.

        Args:
            user_query: Customer's order request or response
            chat_history: Optional list of previous messages in conversation

        Returns:
            OrderResponse with structured order status and message
        """
        logger.info(f"Processing order request: '{user_query}'")

        messages = chat_history.copy() if chat_history else []
        messages.append({"role": "user", "content": user_query})

        try:
            result = self.agent.invoke({"messages": messages})
        except Exception as e:
            logger.error(f"Error invoking order agent: {e}", exc_info=True)
            return OrderResponse(
                message="I encountered an error processing your order. Please try again.",
                status="failed",
                missing_fields=[],
            )

        structured_response = result.get("structured_response")

        if not structured_response:
            logger.debug(
                "Agent did not return a structured response - LLM may have had trouble determining intent"
            )
            # Provide a helpful fallback message that guides the user
            return OrderResponse(
                message=(
                    "I'm not quite sure what you'd like to do. Could you clarify?\n"
                    "• If you want to order a product, please provide the product ID (e.g., 'TECH-001') and quantity\n"
                    "• If you want to search for products, I can help you browse our catalog\n"
                    "• If you're continuing an order, please answer my previous question"
                ),
                status="collecting_info",
                missing_fields=[],
            )

        logger.info(f"Order status: {structured_response.status}")
        return structured_response
