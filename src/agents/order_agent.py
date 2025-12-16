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
        timeout: int = 60,
        cart: Optional[List[Dict]] = None,
    ):
        """
        Initialize the Order Agent.

        Args:
            model_name: OpenAI model to use (must support structured output)
            temperature: Sampling temperature (0 = deterministic)
            timeout: Request timeout in seconds (default: 15)
            cart: Reference to orchestrator's cart list for storing items
        """
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.catalog = ProductCatalog()
        self.order_db = OrderDatabase()
        self.cart = cart if cart is not None else []

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
        def add_to_cart(product_id: str, quantity: int = 1) -> str:
            """
            Validate product and add it to the shopping cart.

            This tool validates the product (checks existence, availability, stock)
            and adds it to the cart. If the product is already in the cart, it updates
            the quantity.

            Use this when customer mentions a product they want to order.

            Args:
                product_id: The product ID to add to cart
                quantity: Quantity to add (default: 1)

            Returns:
                Success message with product details, or error message if validation fails
            """
            logger.debug(
                f"Tool called: add_to_cart(product_id='{product_id}', quantity={quantity})"
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

            existing_item = None
            for item in self.cart:
                if item["product_id"] == product_id:
                    existing_item = item
                    break

            if existing_item:
                existing_item["quantity"] = quantity
                subtotal = product["price"] * quantity
                result = (
                    f"✓ Updated cart:\n"
                    f"- {product['name']} (ID: {product_id})\n"
                    f"- Quantity: {quantity}\n"
                    f"- Subtotal: ${subtotal:.2f}\n"
                )
            else:
                self.cart.append(
                    {
                        "product_id": product_id,
                        "product_name": product["name"],
                        "quantity": quantity,
                        "unit_price": product["price"],
                    }
                )
                subtotal = product["price"] * quantity
                result = (
                    f"✓ Added to cart:\n"
                    f"- {product['name']} (ID: {product_id})\n"
                    f"- Quantity: {quantity}\n"
                    f"- Price: ${product['price']:.2f} each\n"
                    f"- Subtotal: ${subtotal:.2f}\n"
                    f"- Stock: {stock_status.replace('_', ' ').title()}\n"
                )

            if stock_status == "low_stock":
                result += "\nNote: This item has limited stock available."

            return result

        @tool
        def remove_from_cart(product_id: str) -> str:
            """
            Remove an item from the shopping cart.

            Use this when the customer wants to remove a product from their cart.

            Args:
                product_id: The product ID to remove from cart

            Returns:
                Confirmation message if removed, or error if not found in cart
            """
            logger.debug(f"Tool called: remove_from_cart(product_id='{product_id}')")

            removed = False
            product_name = None
            for i, item in enumerate(self.cart):
                if item["product_id"] == product_id:
                    product_name = item["product_name"]
                    self.cart.pop(i)
                    removed = True
                    break

            if removed:
                return f"✓ Removed {product_name} (ID: {product_id}) from your cart."
            else:
                return f"Product {product_id} is not in your cart."

        @tool
        def view_cart() -> str:
            """
            View the current contents of the shopping cart.

            Use this when customer asks about their cart, wants to see what's in it,
            or wants to review items before checkout.

            Returns:
                Formatted cart contents with items, quantities, prices, and total
            """
            logger.debug("Tool called: view_cart()")

            if not self.cart:
                return "Your cart is empty. Add some products to get started!"

            lines = []
            total = 0

            for item in self.cart:
                subtotal = item["quantity"] * item["unit_price"]
                total += subtotal
                lines.append(
                    f"- {item['quantity']}x {item['product_name']} (ID: {item['product_id']}) "
                    f"@ ${item['unit_price']:.2f} each = ${subtotal:.2f}"
                )

            result = "🛒 Your Cart:\n\n" + "\n".join(lines) + f"\n\nTotal: ${total:.2f}"

            return result

        @tool
        def create_order(
            customer_name: str,
            email: str,
            shipping_address: str,
        ) -> str:
            """
            Create a new order from the shopping cart.

            Use this ONLY after you have:
            1. Added all desired products to cart using add_to_cart
            2. Collected ALL customer information (name, email, address)
            3. Confirmed the order with the customer

            The order will be created from the current cart contents. The cart will be
            cleared after successful order creation.

            Args:
                customer_name: Customer's full name
                email: Customer's email address
                shipping_address: Complete shipping address

            Returns:
                Order confirmation with order ID and details
            """

            logger.debug(f"Tool called: create_order")

            if not self.cart:
                return "Error: Your cart is empty. Please add items to your cart before placing an order."

            order_items = []
            total = 0
            items_summary = []

            for cart_item in self.cart:
                product_id = cart_item["product_id"]
                quantity = cart_item["quantity"]

                product = self.catalog.get_product(product_id)
                if not product:
                    return f"Error: Product {product_id} not found. It may have been removed from the catalog."

                if not self.catalog.is_available(product_id):
                    return f"Error: {product['name']} is now out of stock. Please remove it from your cart or choose an alternative."

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

            self.cart.clear()

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
            "CRITICAL: ANTI-HALLUCINATION RULES\n"
            "1. If the user EXPLICITLY provides a Product ID in their message (e.g., 'Buy SPORT-003'), TRUST IT and use it with add_to_cart.\n"
            "   - The add_to_cart tool will validate if the ID exists. If it fails, report the error to the user.\n"
            "2. If the user refers to a product BY NAME ONLY (e.g., 'Buy the yoga mat', 'Add AirPods'):\n"
            "   - Look in chat history for the Product ID associated with that name.\n"
            "   - If found (e.g., 'AirPods Pro (ID: TECH-003)'), use that exact ID.\n"
            "   - If NOT found, DO NOT GUESS. Ask: 'Could you provide the Product ID or would you like me to search for it?'\n"
            "3. NEVER invent a Product ID. Only use IDs that are either:\n"
            "   - Explicitly stated by the user in the current message, OR\n"
            "   - Found in chat history with a matching product name.\n"
            "4. If add_to_cart fails, report the exact error to the user and ask for clarification.\n"
            "\n"
            "CRITICAL: MANDATORY ORDER CONFIRMATION\n"
            "1. You MUST ask 'Are you ready to place your order?' BEFORE calling create_order.\n"
            "2. NEVER call create_order immediately after receiving the shipping address.\n"
            "3. The correct sequence is: collect address → show order summary with view_cart → ask 'Are you ready?' → wait for 'yes' → create_order\n"
            "4. Only call create_order when the customer explicitly says 'yes', 'confirm', 'place order', or similar AFTER you asked for confirmation.\n"
            "5. Providing the address is NOT consent to place the order. You MUST still ask for explicit confirmation.\n"
            "\n"
            "USING CHAT HISTORY:\n"
            "- The conversation history is available in the messages you receive\n"
            "- To find product_id when customer mentions product by name:\n"
            "  1. Look through previous assistant messages for product listings\n"
            "  2. Search for patterns like 'Product ID: TECH-001' or '(ID: TECH-001)' or '**Product ID:** TECH-001'\n"
            "  3. Match the product name mentioned by customer to the name in previous listings\n"
            "  4. Extract the EXACT product_id from that listing - do not modify or guess it\n"
            "- Example: If customer says 'I want the macbook' and earlier you see 'MacBook Pro 16-inch (ID: TECH-001)', use TECH-001\n"
            "- If product_id cannot be found in chat history, DO NOT GUESS - ask customer to provide the product ID\n"
            "\n"
            "TOOLS AVAILABLE:\n"
            "1. add_to_cart - Validate product and add/update it in the shopping cart\n"
            "2. remove_from_cart - Remove an item from the shopping cart\n"
            "3. view_cart - Show current cart contents with items, quantities, and total\n"
            "4. create_order - Create order from cart (requires customer info: name, email, address)\n"
            "5. transfer_to_rag_agent - Transfer customer back to product search\n"
            "\n"
            "WHEN TO TRANSFER:\n"
            "- Customer wants to search for products\n"
            "- Customer wants to browse catalog or get product info\n"
            "- Customer wants product recommendations or comparisons\n"
            "When transferring: Use transfer_to_rag_agent tool, set 'transfer_to_agent' field to 'rag', and include a friendly message.\n"
            "\n"
            "ORDER PROCESS:\n"
            "1. ADD TO CART: Use add_to_cart tool for EACH product_id and quantity\n"
            "   - If user provides a Product ID (e.g., 'order TECH-009'), USE IT DIRECTLY with add_to_cart\n"
            "   - If user mentions product BY NAME ONLY, look in chat history for the ID\n"
            "   - Only ask for ID if user gives a name AND it's not in chat history\n"
            "   - Extract quantity from customer's message:\n"
            "     * Numbers: 'I want 2 macbook' → quantity = 2\n"
            "     * Number words: 'I want three laptops' → quantity = 3\n"
            "     * Articles: 'I want to buy a macbook' → quantity = 1\n"
            "     * No quantity mentioned: 'I want macbook' → quantity = 1 (infer from context)\n"
            "   - If quantity is clear (explicit number or article), add to cart immediately\n"
            "   - If quantity is truly ambiguous, ask 'How many would you like to order?'\n"
            "   - add_to_cart validates AND stores - no separate validation step needed\n"
            "2. ASK TO ADD MORE: After each add_to_cart, ask 'Would you like to add anything else to your cart?'\n"
            "3. VIEW CART: Use view_cart tool when customer asks about their cart or before checkout\n"
            "4. COLLECT INFO: Once done shopping, collect - Name, Email, Full shipping address\n"
            "5. FINAL CONFIRMATION: Use view_cart to show summary, then EXPLICITLY ask: 'Are you ready to place your order?'\n"
            "6. CHECKOUT: ONLY when user says 'yes' or 'place order' to the final confirmation, use create_order\n"
            "   - DO NOT call create_order just because you have the address. You MUST get a final 'yes'.\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- Extract quantity from customer's message FIRST before asking questions\n"
            "- Use add_to_cart to add/update items - it validates AND stores in one step\n"
            "- Use remove_from_cart when customer wants to remove an item from cart\n"
            "- Use view_cart whenever customer asks 'what's in my cart?' or 'show my cart'\n"
            "- To update quantity, use add_to_cart again with new quantity (replaces old quantity)\n"
            "- Use ONLY product information from add_to_cart tool results - never invent prices, names, or details\n"
            "- If add_to_cart returns an error (product not found, out of stock), use that exact information\n"
            "- After each add_to_cart, ask about adding more items\n"
            "- Support multiple items in a single order\n"
            "- Never create order without FINAL confirmation ('Are you ready to place your order?')\n"
            "- Sending the address is NOT a confirmation to place the order immediately. You must still ask 'Are you ready?'\n"
            "- Handle out-of-stock by offering alternatives or transferring to search\n"
            "- Ask for one detail at a time if not all provided\n"
            "- create_order automatically uses cart contents - no need to pass items\n"
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
            tools=[
                transfer_to_rag_agent,
                add_to_cart,
                remove_from_cart,
                view_cart,
                create_order,
            ],
            system_prompt=system_prompt,
            response_format=OrderResponse,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=10,
                    exit_behavior="end",
                ),
                ToolCallLimitMiddleware(
                    tool_name="create_order",
                    run_limit=1,
                ),
                ToolCallLimitMiddleware(
                    tool_name="view_cart",
                    run_limit=2,
                ),
                ToolCallLimitMiddleware(
                    tool_name="add_to_cart",
                    run_limit=3,
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
