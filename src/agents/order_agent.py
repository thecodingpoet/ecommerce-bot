"""
Order Agent for handling purchase requests and order creation.
Uses LangChain's agent pattern with tools for order operations.
"""

import json
import logging
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from database import OrderDatabase, ProductCatalog

load_dotenv()

logger = logging.getLogger(__name__)


class OrderResponse(BaseModel):
    """Structured response from Order Agent."""

    message: str = Field(description="Natural language response to customer")
    status: Literal["collecting_info", "confirming", "completed", "failed"] = Field(
        description="Current status of order process"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of missing required fields (e.g., 'quantity', 'email')",
    )
    order_summary: Optional[dict] = Field(
        None, description="Order summary when ready to confirm"
    )
    order_id: Optional[str] = Field(
        None, description="Order ID after successful creation"
    )
    transfer_to_agent: Optional[str] = Field(
        None,
        description="Agent to transfer to: 'rag' for product search, or None to continue with Order Agent",
    )


class OrderAgent:
    """
    Order Agent for processing customer orders.

    Uses LangChain's agent pattern with tools for product validation
    and order creation.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize the Order Agent.

        Args:
            model_name: OpenAI model to use (must support structured output)
            temperature: Sampling temperature (0 = deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
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

            # Validate all products and build order items
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

            # Create order
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

        model = ChatOpenAI(model=model_name, temperature=temperature)

        system_prompt = (
            "You are a helpful order assistant for an e-commerce store. Your role is to help customers place orders. "
            "\n\n"
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
            "   - Customer MUST provide exact product_id (e.g., 'TECH-001', 'HOME-004')\n"
            "   - If customer says product name without ID (e.g., 'MacBook'), politely ask for the product ID\n"
            "   - Example: 'I'd be happy to help you order the MacBook Pro! Could you please provide the product ID? You can find it in the product listing above.'\n"
            "   - ALWAYS CLARIFY QUANTITY: If customer doesn't specify quantity, ask 'How many would you like to order?' before validation\n"
            "   - If customer says 'I want product X' without quantity, assume they might want 1 but ALWAYS confirm: 'Would you like to order 1 unit, or a different quantity?'\n"
            "2. ASK TO ADD MORE: After each validation, ask 'Would you like to add anything else to your order?'\n"
            "3. COLLECT INFO: Once done shopping, collect - Name, Email, Full shipping address\n"
            "4. CONFIRM: Show summary with ALL items (with quantities) and total, ask for final confirmation\n"
            "5. CREATE: Use create_order with ALL items as JSON array\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- Look in chat history to find product_id when customer mentions product by name\n"
            "- ALWAYS clarify and confirm the quantity for each item before validation\n"
            "- If quantity is not mentioned, explicitly ask 'How many would you like?' before proceeding\n"
            "- Validate ALL products before collecting customer info\n"
            "- After each validation, ask about adding more items\n"
            "- Support multiple items in a single order\n"
            '- Items format: \'[{"product_id": "TECH-007", "quantity": 2}]\'\n'
            "- Never create order without confirmation\n"
            "- Handle out-of-stock by offering alternatives or transferring to search\n"
            "- Ask for one detail at a time if not all provided\n"
            "\n"
            "RESPONSE FORMAT - CRITICAL:\n"
            "- 'status': MUST be one of: 'collecting_info', 'confirming', 'completed', 'failed'\n"
            "  * 'collecting_info': Asking for product ID, validating products, collecting customer details (name/email/address), or asking about adding more items\n"
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
                    run_limit=15,
                    exit_behavior="end",
                )
            ],
        )

        logger.info(
            f"Order Agent initialized with model={model_name}, temperature={temperature}"
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
            logger.error(f"Error invoking order agent: {e}")
            return OrderResponse(
                message="I encountered an error processing your order. Please try again.",
                status="failed",
                missing_fields=[],
            )

        structured_response = result.get("structured_response")

        if not structured_response:
            logger.error("Agent did not return a structured response")
            return OrderResponse(
                message="I'm having trouble processing your order. Please try again.",
                status="failed",
                missing_fields=[],
            )

        logger.info(f"Order status: {structured_response.status}")
        return structured_response
