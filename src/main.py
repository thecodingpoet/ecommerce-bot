"""
E-commerce chatbot CLI.
Interactive assistant for product search and order placement.
"""

import argparse
import logging
from enum import Enum

from dotenv import load_dotenv

from agents.order_agent import OrderAgent
from agents.rag_agent import RAGAgent
from utils.logger import setup_logger
from utils.spinner import Spinner

load_dotenv()


class Agent(Enum):
    """Available agent types."""

    RAG = "rag"
    ORDER = "order"

    @classmethod
    def default(cls):
        """Return the default agent."""
        return cls.RAG


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level.

    Args:
        verbose: Enable DEBUG level logging if True, otherwise INFO

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.getLogger().setLevel(log_level)

    return setup_logger("ecommerce-cli", level=log_level)


def print_banner(verbose: bool = False):
    """Print welcome banner with application info and commands.

    Args:
        verbose: Whether verbose mode is enabled
    """
    print("\n" + "=" * 70)
    print("🛍️  Welcome to Our E-Commerce Store!")
    print("=" * 70)
    print("I'm your AI shopping assistant. I can help you:")
    print("  • Search and browse products")
    print("  • Get detailed product information")
    print("  • Place orders with ease")
    print()
    print("Commands:")
    print("  • Type 'exit' or 'quit' to end the conversation")
    if verbose:
        print("  • Verbose mode is enabled - debug info will be shown")
    print("=" * 70)


def handle_rag_response(response, logger):
    """Handle RAG agent response and check for transfers.

    Args:
        response: RAG agent response object
        logger: Logger instance

    Returns:
        Tuple of (next_agent, message) where next_agent is Agent enum or None
    """
    print(f"\nAssistant: {response.answer}")

    next_agent = None
    if response.transfer_to_agent == Agent.ORDER.value:
        next_agent = Agent.ORDER
        logger.info("Transferring control to Order Agent")
        print("\n[Connecting you with our order specialist...]")

    return next_agent, response.answer


def handle_order_response(response, logger):
    """Handle Order agent response and check for transfers.

    Args:
        response: Order agent response object
        logger: Logger instance

    Returns:
        Tuple of (next_agent, message, reset_history)
    """
    print(f"\nAssistant: {response.message}")

    next_agent = None
    reset_history = False

    if response.status == "completed":
        print(f"\n✅ Order completed! Order ID: {response.order_id}")
        logger.info("Resetting chat history after order completion")
        reset_history = True
        next_agent = Agent.RAG
    elif response.status == "failed":
        print("\n❌ Order failed.")
        logger.info("Resetting chat history after order failure")
        reset_history = True
        next_agent = Agent.RAG

    if response.transfer_to_agent == Agent.RAG.value:
        next_agent = Agent.RAG
        logger.info("Transferring control to RAG Agent")
        print("\n[Returning to product search...]")

    return next_agent, response.message, reset_history


def invoke_agent(
    current_agent, user_input, chat_history, rag_agent, order_agent, logger
):
    """Invoke the appropriate agent and handle response.

    Args:
        current_agent: Current Agent enum
        user_input: User's input message
        chat_history: Conversation history
        rag_agent: RAG agent instance
        order_agent: Order agent instance
        logger: Logger instance

    Returns:
        Tuple of (next_agent, message, reset_history)
    """
    logger.debug(f"Current agent: {current_agent.name}")
    logger.debug(f"Chat history length: {len(chat_history)}")

    spinner = Spinner("Processing")
    spinner.start()

    try:
        if current_agent == Agent.RAG:
            response = rag_agent.invoke(user_input, chat_history=chat_history)
            next_agent, message = handle_rag_response(response, logger)
            return next_agent, message, False
        else:
            response = order_agent.invoke(user_input, chat_history=chat_history)
            return handle_order_response(response, logger)
    finally:
        spinner.stop()


def main(verbose: bool = False):
    """Main CLI - E-commerce assistant with product search and ordering.

    Args:
        verbose: Enable verbose output for debugging
    """
    # Configure logging FIRST, before creating agents
    logger = setup_logging(verbose)

    rag_agent = RAGAgent()
    order_agent = OrderAgent()
    chat_history = []
    current_agent = Agent.default()

    print_banner(verbose)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for shopping with us! Goodbye!")
            break

        if not user_input:
            continue

        try:
            next_agent, message, reset_history = invoke_agent(
                current_agent, user_input, chat_history, rag_agent, order_agent, logger
            )

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": message})

            if reset_history:
                chat_history = []
            if next_agent:
                current_agent = next_agent

        except KeyboardInterrupt:
            print("\n\nThank you for shopping with us! Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            if verbose:
                logger.exception("Full traceback:")
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-commerce chatbot CLI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging",
    )
    args = parser.parse_args()

    main(verbose=args.verbose)
