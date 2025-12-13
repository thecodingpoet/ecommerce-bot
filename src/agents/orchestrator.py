import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.order_agent import OrderAgent
from agents.rag_agent import RAGAgent

load_dotenv()

logger = logging.getLogger(__name__)


class OrchestratorResponse(BaseModel):
    """Structured response from Orchestrator."""

    message: str = Field(description="Response to the user")
    agent_used: str = Field(
        description="Which agent handled the request: 'rag', 'order', or 'orchestrator'"
    )


class Orchestrator:
    """
    Orchestrator agent that coordinates between RAG and Order agents.

    Routes user queries to the appropriate specialized agent:
    - Product search queries → RAG Agent
    - Order/purchase requests → Order Agent
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize the Orchestrator.

        Args:
            model_name: OpenAI model to use
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.temperature = temperature
        self._chat_history = []
        self._in_order_mode = False

        self.rag_agent = RAGAgent(model_name=model_name, temperature=temperature)
        self.order_agent = OrderAgent(model_name=model_name, temperature=temperature)

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

            self._in_order_mode = False

            result = self.rag_agent.invoke(request, chat_history=self._chat_history)

            response = result.answer
            if result.products:
                response += f"\n\n[Retrieved {len(result.products)} products]"

            return response

        @tool
        def place_order(request: str) -> str:
            """
            Handle order placement and shopping cart management.

            Use this when the user wants to:
            - Place an order or buy a product
            - Add items to cart
            - Checkout
            - Provide shipping/billing information
            - Confirm a purchase

            The order agent will guide the user through:
            - Product validation
            - Collecting customer information (name, email, address)
            - Order confirmation
            - Order creation

            Input: Natural language order request
            (e.g., 'I want to buy TECH-007', 'order 2 laptops')
            """
            logger.info(f"Routing to Order Agent: {request}")

            self._in_order_mode = True

            result = self.order_agent.invoke(request, chat_history=self._chat_history)

            if result.status == "completed" or result.transfer_to_agent == "rag":
                self._in_order_mode = False
                logger.info(f"Order {result.status}, exiting order mode")

                if result.transfer_to_agent == "rag":
                    return (
                        "Let me transfer you back to product search. " + result.message
                    )

            return result.message

        model = ChatOpenAI(model=model_name, temperature=temperature)

        system_prompt = (
            "You are a helpful e-commerce assistant that helps customers find and purchase products. "
            "You have two specialized capabilities: "
            "1. search_products - for finding and learning about products in the catalog "
            "2. place_order - for purchasing products and managing orders "
            "\n\n"
            "ROUTING RULES: "
            "- Use search_products when users ask about products, features, pricing, availability, or want to browse "
            "- Use place_order ONLY ONCE when users first express intent to buy/purchase/order "
            "- After calling place_order, DO NOT call it again - the order agent will handle the conversation "
            "- For greetings (hi, hello, hey), respond warmly and ask how you can help with products or orders "
            "- For questions unrelated to products or orders (weather, news, general knowledge, etc.), politely decline: "
            "  'I'm specialized in helping you find and purchase products. I can search our catalog or help you place an order. "
            "  For other questions, please contact our customer support team. What products can I help you with today?' "
            "- If the request is ambiguous, ask clarifying questions "
            "- Always use tools for product/order requests - do not try to answer about products without using search_products "
            "- Be friendly, concise, and helpful"
        )

        self.agent = create_agent(
            model,
            tools=[search_products, place_order],
            system_prompt=system_prompt,
            response_format=OrchestratorResponse,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=10,
                    exit_behavior="end",
                )
            ],
        )

        logger.info(
            f"Orchestrator initialized with model={model_name}, temperature={temperature}"
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
            f"Orchestrator processing: '{user_query}' (order_mode={self._in_order_mode})"
        )

        self._chat_history = chat_history.copy() if chat_history else []

        if self._in_order_mode:
            logger.info("In order mode, routing directly to order agent")
            result = self.order_agent.invoke(
                user_query, chat_history=self._chat_history
            )

            if result.status == "completed" or result.transfer_to_agent == "rag":
                self._in_order_mode = False
                logger.info(f"Order {result.status}, exiting order mode")

                if result.transfer_to_agent == "rag":
                    logger.info("Order agent requested transfer to RAG, routing query")
                    rag_result = self.rag_agent.invoke(
                        user_query, chat_history=self._chat_history
                    )
                    return OrchestratorResponse(
                        message=rag_result.answer, agent_used="rag"
                    )

            return OrchestratorResponse(message=result.message, agent_used="order")

        messages = self._chat_history.copy()
        messages.append({"role": "user", "content": user_query})

        try:
            result = self.agent.invoke({"messages": messages})
        except Exception as e:
            logger.error(f"Error invoking orchestrator: {e}")
            return OrchestratorResponse(
                message="I encountered an error processing your request. Please try again.",
                agent_used="orchestrator",
            )

        structured_response = result.get("structured_response")

        if not structured_response:
            logger.error("Orchestrator did not return a structured response")
            return OrchestratorResponse(
                message="I'm having trouble processing your request. Please try again.",
                agent_used="orchestrator",
            )

        logger.info(f"Agent used: {structured_response.agent_used}")
        return structured_response
