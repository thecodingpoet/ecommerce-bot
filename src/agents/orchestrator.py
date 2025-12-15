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

    - SEARCH: Default state, routes queries to appropriate agent
    - ORDER_LOCKED: Locked to Order Agent during active order process
    """

    SEARCH = "search"
    ORDER_LOCKED = "order_locked"

    def is_order_mode(self) -> bool:
        """Check if currently in order mode."""
        return self == OrchestratorState.ORDER_LOCKED

    @classmethod
    def should_exit_order_mode(
        cls, order_status: str, transfer_to_agent: Optional[str]
    ) -> bool:
        """
        Determine if we should exit ORDER_LOCKED state based on order agent response.

        Args:
            order_status: OrderAgent status ("collecting_info", "confirming", "completed", "failed")
            transfer_to_agent: Transfer request from agent ("rag", "order", or None)

        Returns:
            True if should exit order mode (transition to SEARCH)
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
        timeout: int = 15,
    ):
        """
        Initialize the Orchestrator.

        Args:
            model_name: OpenAI model to use
            temperature: Sampling temperature
            timeout: Request timeout in seconds (default: 15)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self._chat_history = []
        self._state = OrchestratorState.SEARCH
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

            self._state = OrchestratorState.SEARCH

            result = self.rag_agent.invoke(request, chat_history=self._chat_history)

            response = result.message
            if result.products:
                response += f"\n\n[Retrieved {len(result.products)} products]"

            return response

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

            self._state = OrchestratorState.ORDER_LOCKED

            result = self.order_agent.invoke(request, chat_history=self._chat_history)

            if OrchestratorState.should_exit_order_mode(
                result.status, result.transfer_to_agent
            ):
                self._state = OrchestratorState.SEARCH
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
            "ROUTING RULES: "
            "- Use search_products when users ask about products, features, pricing, availability, or want to browse "
            "- Use manage_order ONLY ONCE when users first express intent to buy/purchase/order "
            "- Use manage_order when users ask to 'view cart', 'check cart', 'see my items', or 'checkout' "
            "- After calling manage_order, DO NOT call it again - the order agent will handle the conversation "
            "- For greetings (hi, hello, hey), respond warmly and ask how you can help with products or orders "
            "\n\n"
            "HANDLING AMBIGUOUS OR UNCLEAR QUERIES: "
            "- If the query is ambiguous, unclear, or you cannot determine which agent should handle it, "
            "  DO NOT call any tool. Instead, respond directly with clarifying questions. "
            "- Examples of ambiguous queries: single numbers (e.g., '2'), vague terms, incomplete sentences, "
            "  or queries that could refer to either product search or ordering "
            "- Ask specific questions to understand the user's intent, such as: "
            "  'I'd like to help you, but could you clarify what you're looking for? Are you trying to: "
            "  (1) search for products, or (2) place an order? If ordering, do you have a specific product ID?' "
            "\n\n"
            "OUT-OF-SCOPE QUERIES: "
            "- For questions unrelated to products or orders (weather, news, general knowledge, etc.), "
            "  DO NOT call any tool. Politely decline directly: "
            "'I'm specialized in helping you find and purchase products. I can search our catalog or help you place an order. "
            "For other questions, please contact our customer support team. What products can I help you with today?' "
            "\n\n"
            "GENERAL GUIDELINES: "
            "- Always use tools for product/order requests - do not try to answer about products without using search_products "
            "- Be friendly, concise, and helpful"
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

        if self._state.is_order_mode():
            logger.info("In order mode, routing directly to order agent")
            result = self.order_agent.invoke(
                user_query, chat_history=self._chat_history
            )

            if OrchestratorState.should_exit_order_mode(
                result.status, result.transfer_to_agent
            ):
                self._state = OrchestratorState.SEARCH
                logger.info(f"Order {result.status}, exiting order mode")

                if result.transfer_to_agent == "rag":
                    logger.info("Order agent requested transfer to RAG, routing query")
                    rag_result = self.rag_agent.invoke(
                        user_query, chat_history=self._chat_history
                    )

                    # Check if RAG agent bounced it back to order
                    # This happens if the user query was "add more items" (triggering transfer to RAG)
                    # but RAG sees it as an order intent and wants to transfer back.
                    # In this case, we should just show the Order agent's transition message
                    # and let the user enter their actual search query next.
                    if rag_result.transfer_to_agent == "order":
                        logger.info(
                            "RAG agent bounced back (not a search query). Returning order agent transition message."
                        )
                        return OrchestratorResponse(
                            message=result.message, agent_used="order"
                        )

                    return OrchestratorResponse(
                        message=rag_result.message, agent_used="rag"
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
