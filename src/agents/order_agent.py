"""
Order Agent for handling purchase requests and order creation.
Uses LangChain's agent pattern with tools for order operations.
"""

import json
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from database import OrderDatabase, ProductCatalog
from utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)


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
                f"You'll receive a confirmation email at {email}.\n"
                f"Estimated delivery: 3-5 business days."
            )

        model = ChatOpenAI(model=model_name, temperature=temperature)

        system_prompt = (
            "You are a helpful order assistant for an e-commerce store. Your role is to help customers place orders. "
            "When a customer wants to order something, follow these steps: "
            "1. VALIDATE PRODUCTS: Use validate_product tool for EACH product_id and quantity the customer wants. "
            "2. ASK TO ADD MORE: After validating each product, ask 'Would you like to add anything else to your order?' before collecting customer info. "
            "3. COLLECT CUSTOMER INFO: Once customer is done adding items, collect ALL of these - Customer's full name, Email address, Complete shipping address (street, city, state, zip). "
            "4. CONFIRM ORDER: Show a summary with ALL items and total price, ask for confirmation. "
            "5. CREATE ORDER: Once confirmed, use create_order tool with ALL validated items as a JSON array. "
            "IMPORTANT RULES: "
            "Always validate ALL products before collecting customer info. "
            "After each product validation, ask if they want to add more items. "
            "Only proceed to collect customer info when they confirm they're done shopping. "
            "Support multiple items in a single order - keep track of all validated products. "
            'When calling create_order, pass items as a JSON string array like: \'[{"product_id": "TECH-007", "quantity": 2}, {"product_id": "HOME-004", "quantity": 1}]\'. '
            "Never create an order without confirming with the customer first. "
            "If any product is out of stock, inform customer and offer alternatives. "
            "Be conversational and friendly. "
            "Ask for one piece of information at a time if customer doesn't provide all details. "
            "Use the 'missing_fields' array to track what info you still need. "
            "In your structured response: "
            "Set status to 'collecting_info' while gathering details or asking about more items. "
            "Set status to 'confirming' when showing order summary. "
            "Set status to 'completed' after successful order creation. "
            "Set status to 'failed' if order cannot be completed. "
            "List any missing_fields that still need to be collected."
        )

        self.agent = create_agent(
            model,
            tools=[validate_product, create_order],
            system_prompt=system_prompt,
            response_format=OrderResponse,
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

        result = self.agent.invoke({"messages": messages})

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
