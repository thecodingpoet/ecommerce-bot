"""
E-commerce chatbot CLI.
Interactive assistant for product search and order placement.
"""

import argparse
import logging

from dotenv import load_dotenv

from agents.orchestrator import Orchestrator
from utils.logger import setup_logger
from utils.spinner import Spinner

load_dotenv()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for all components.

    Args:
        verbose: Enable verbose logging if True

    Returns:
        Logger instance for CLI
    """
    log_level = logging.DEBUG if verbose else logging.WARNING

    for component in [
        "agents.order_agent",
        "agents.orchestrator",
        "agents.rag_agent",
        "database.products",
    ]:
        setup_logger(component, level=log_level)

    return setup_logger("cli", level=log_level)


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


def main(verbose: bool = False):
    """Main CLI - E-commerce assistant with product search and ordering.

    Args:
        verbose: Enable verbose output for debugging
    """
    logger = setup_logging(verbose)

    orchestrator = Orchestrator()
    chat_history = []

    print_banner(verbose)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nThank you for shopping with us! Goodbye!")
            break

        if not user_input:
            continue

        try:
            spinner = Spinner("Processing")
            spinner.start()

            try:
                logger.debug(f"Order mode: {orchestrator._in_order_mode}")
                logger.debug(f"Chat history length: {len(chat_history)}")

                response = orchestrator.invoke(user_input, chat_history=chat_history)
                response_message = response.message

            finally:
                spinner.stop()

            print(f"Assistant: {response_message}")

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response_message})

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
