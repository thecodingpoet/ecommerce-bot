import logging
from enum import Enum
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agents.order_agent import OrderAgent
from agents.rag_agent import RAGAgent
from schema import OrchestratorResponse

load_dotenv()

logger = logging.getLogger(__name__)


class OrchestratorState(str, Enum):
    """
    Orchestrator conversation states.

    - INTENT: Default state, routes queries to appropriate agent
    - CHECKOUT: Locked to Order Agent during active order process
    """

    INTENT = "intent"
    CHECKOUT = "checkout"

    def is_checkout_mode(self) -> bool:
        """Check if currently in checkout mode."""
        return self == OrchestratorState.CHECKOUT

    @classmethod
    def should_exit_checkout_mode(
        cls, order_status: str, transfer_to_agent: Optional[str]
    ) -> bool:
        """
        Determine if we should exit CHECKOUT state based on order agent response.

        Args:
            order_status: OrderAgent status ("collecting_info", "confirming", "completed", "failed")
            transfer_to_agent: Transfer request from agent ("rag", "order", or None)

        Returns:
            True if should exit checkout mode (transition to INTENT)
        """
        if order_status in ("completed", "failed"):
            return True

        if transfer_to_agent == "rag":
            return True

        return False


class Orchestrator:
    """
    Orchestrator agent that coordinates between RAG and Order agents.

    Routes user queries to the appropriate specialized agent:
    - Product search queries → RAG Agent
    - Order/purchase requests → Order Agent
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        timeout: int = 60,
    ):
        """
        Initialize the Orchestrator.

        Args:
            model_name: OpenAI model to use
            temperature: Sampling temperature
            timeout: Request timeout in seconds (default: 60)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self._chat_history = []
        self._state = OrchestratorState.INTENT
        self._cart = []

        self.rag_agent = RAGAgent(
            model_name=model_name, temperature=temperature, timeout=timeout
        )
        self.order_agent = OrderAgent(
            model_name=model_name,
            temperature=temperature,
            timeout=timeout,
            cart=self._cart,
        )

        @tool
        def search_products(request: str) -> str:
            """
            Search for products in the catalog using natural language.

            Use this when the user wants to:
            - Find products by name, category, or features
            - Browse available products
            - Get product information, specifications, or pricing
            - Compare products
            - Ask about product availability

            Input: Natural language product search query
            (e.g., 'show me wireless headphones', 'what laptops do you have?')
            """
            logger.info(f"Routing to RAG Agent: {request}")

            self._state = OrchestratorState.INTENT

            result = self.rag_agent.invoke(request, chat_history=self._chat_history)

            return self._append_product_details(result.message, result.products)

        @tool
        def manage_order(request: str) -> str:
            """
            Handle order placement and shopping cart management.

            Use this when the user wants to:
            - Place an order or buy a product
            - Add items to cart
            - View or check shopping cart
            - Checkout
            - Provide shipping/billing information
            - Confirm a purchase

            The order agent will guide the user through:
            - Product validation
            - Collecting customer information (name, email, address)
            - Order confirmation
            - Order creation

            Input: Natural language order request
            (e.g., 'I want to buy TECH-007', 'order 2 laptops', 'show my cart')
            """
            logger.info(f"Routing to Order Agent: {request}")

            self._state = OrchestratorState.CHECKOUT

            result = self.order_agent.invoke(request, chat_history=self._chat_history)

            if OrchestratorState.should_exit_checkout_mode(
                result.status, result.transfer_to_agent
            ):
                self._state = OrchestratorState.INTENT
                logger.info(f"Order {result.status}, exiting order mode")

                if result.transfer_to_agent == "rag":
                    return (
                        "Let me transfer you back to product search. " + result.message
                    )

            return result.message

        model = ChatOpenAI(model=model_name, temperature=temperature, timeout=timeout)

        system_prompt = (
            "You are a helpful e-commerce assistant that helps customers find and purchase products. "
            "You have two specialized capabilities: "
            "1. search_products - for finding and learning about products in the catalog "
            "2. manage_order - for purchasing products and managing orders "
            "\n\n"
            "CRITICAL: ANTI-HALLUCINATION RULES\n"
            "1. NEVER answer product questions without calling search_products first. This includes:\n"
            "   - Product names, descriptions, specifications, or features\n"
            "   - Prices or availability\n"
            "   - Comparisons between products\n"
            "   - Follow-up questions about previously mentioned products\n"
            "2. The search_products tool is your SINGLE SOURCE OF TRUTH for product information.\n"
            "3. Even if the user refers to a product mentioned earlier in the conversation, you MUST call search_products to verify current details.\n"
            "4. NEVER invent product IDs, prices, specs, or other details from memory or chat history.\n"
            "5. If you're about to provide product information without calling search_products, STOP. Call the tool instead.\n"
            "\n\n"
            "ROUTING RULES:\n"
            "- Use search_products for ANY product-related query including:\n"
            "  - 'Tell me more about X' or 'more details on X'\n"
            "  - 'How does X compare to Y?' or 'which is better?'\n"
            "  - 'What's the price of X?'\n"
            "  - Follow-up questions like 'what about the other one?'\n"
            "- Use manage_order ONLY ONCE when users first express intent to buy/purchase/order\n"
            "- Use manage_order when users ask to 'view cart', 'check cart', 'see my items', or 'checkout'\n"
            "- After calling manage_order, DO NOT call it again - the order agent will handle the conversation\n"
            "- For greetings (hi, hello, hey), respond warmly and ask how you can help with products or orders\n"
            "\n\n"
            "HANDLING AMBIGUOUS OR UNCLEAR QUERIES:\n"
            "- If the query is ambiguous, unclear, or you cannot determine which agent should handle it, "
            "  DO NOT call any tool. Instead, respond directly with clarifying questions.\n"
            "- Examples of ambiguous queries: single numbers (e.g., '2'), vague terms, incomplete sentences, "
            "  or queries that could refer to either product search or ordering\n"
            "- Ask specific questions to understand the user's intent, such as: "
            "  'I'd like to help you, but could you clarify what you're looking for? Are you trying to: "
            "  (1) search for products, or (2) place an order? If ordering, do you have a specific product ID?'\n"
            "\n\n"
            "OUT-OF-SCOPE QUERIES:\n"
            "- For questions unrelated to products or orders (weather, news, general knowledge, etc.), "
            "  DO NOT call any tool. Politely decline directly: "
            "'I'm specialized in helping you find and purchase products. I can search our catalog or help you place an order. "
            "For other questions, please contact our customer support team. What products can I help you with today?'\n"
            "\n\n"
            "GENERAL GUIDELINES:\n"
            "- ALWAYS use search_products for any product information - answering from chat history alone is HALLUCINATION\n"
            "- Be friendly, concise, and helpful\n"
            "\n\n"
            "RESPONSE FORMAT:\n"
            "- You MUST return a valid JSON object matching the OrchestratorResponse schema.\n"
            "- The JSON must have 'message' and 'agent_used' fields.\n"
            "- DO NOT output any text, markdown, or explanations outside the JSON block.\n"
            "- DO NOT include the ```json ... ``` markdown code fence, just the raw JSON object."
        )

        self.agent = create_agent(
            model,
            tools=[search_products, manage_order],
            system_prompt=system_prompt,
            response_format=OrchestratorResponse,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=3,
                    exit_behavior="end",
                )
            ],
        )

        logger.info(
            f"Orchestrator initialized with model={model_name}, temperature={temperature}, timeout={timeout}s"
        )

    def _append_product_details(self, message: str, products: List) -> str:
        """
        Append structured product details to the message.

        This ensures Product IDs are always present in the chat history for the Order Agent to use.
        """
        if not products:
            return message

        product_lines = []
        for p in products:
            product_lines.append(f"• {p.name} (ID: {p.product_id}) - ${p.price:.2f}")

        if product_lines:
            return message + "\n\nProducts Found:\n" + "\n".join(product_lines)

        return message

    def invoke(
        self, user_query: str, chat_history: Optional[List[Dict]] = None
    ) -> OrchestratorResponse:
        """
        Process user query and route to appropriate agent.

        Args:
            user_query: User's question or request
            chat_history: Optional conversation history

        Returns:
            OrchestratorResponse with agent's reply
        """
        logger.info(
            f"Orchestrator processing: '{user_query}' (state={self._state.value})"
        )

        self._chat_history = chat_history.copy() if chat_history else []

        if self._state.is_checkout_mode():
            logger.info("In order mode, routing directly to order agent")
            result = self.order_agent.invoke(
                user_query, chat_history=self._chat_history
            )

            if OrchestratorState.should_exit_checkout_mode(
                result.status, result.transfer_to_agent
            ):
                self._state = OrchestratorState.INTENT
                logger.info(f"Order {result.status}, exiting order mode")

                if result.transfer_to_agent == "rag":
                    logger.info("Order agent requested transfer to RAG, routing query")
                    # Explicitly don't pass history on handover - assume new search intent and avoid timeouts
                    rag_result = self.rag_agent.invoke(user_query, chat_history=[])

                    # If RAG bounces back, return order transition message and await new query.
                    if rag_result.transfer_to_agent == "order":
                        logger.info(
                            "RAG agent bounced back (not a search query). Returning order agent transition message."
                        )
                        return OrchestratorResponse(
                            message=result.message, agent_used="order"
                        )

                    # Append product details to message to preserve IDs in history
                    final_message = self._append_product_details(rag_result.message, rag_result.products)

                    return OrchestratorResponse(
                        message=final_message, agent_used="rag"
                    )

            return OrchestratorResponse(message=result.message, agent_used="order")

        messages = self._chat_history.copy()
        messages.append({"role": "user", "content": user_query})

        try:
            result = self.agent.invoke({"messages": messages})
        except Exception as e:
            logger.error(f"Error invoking orchestrator: {e}", exc_info=True)
            return OrchestratorResponse(
                message="I encountered an error processing your request. Please try again.",
                agent_used="orchestrator",
            )

        structured_response = result.get("structured_response")

        if not structured_response:
            logger.debug(
                "Orchestrator did not return a structured response - LLM may have had trouble determining intent"
            )
            return OrchestratorResponse(
                message=(
                    "I'd like to help you, but could you clarify what you're looking for?\n\n"
                    "Are you trying to:\n"
                    "• Search for products in our catalog (e.g., 'show me laptops' or 'find wireless headphones')\n"
                    "• Place an order (e.g., 'I want to buy TECH-007' or 'order 2 laptops')\n\n"
                    "Please provide more details about what you'd like to do!"
                ),
                agent_used="orchestrator",
            )

        logger.info(f"Agent used: {structured_response.agent_used}")
        return structured_response
