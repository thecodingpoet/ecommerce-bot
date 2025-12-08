"""
E-commerce chatbot CLI.
Interactive assistant for product search and order placement.
"""

import argparse
import logging

from dotenv import load_dotenv

from chatbot import ChatbotSession
from utils.logger import setup_logger
from utils.spinner import Spinner

load_dotenv()


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


def main(verbose: bool = False):
    """Main CLI - E-commerce assistant with product search and ordering.

    Args:
        verbose: Enable verbose output for debugging
    """
    logger = setup_logging(verbose)

    session = ChatbotSession()
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
                logger.debug(f"Order mode: {session.orchestrator._in_order_mode}")
                logger.debug(f"Chat history length: {len(chat_history)}")

                response_message, reset_history, transfer_note = (
                    session.process_message(user_input, chat_history)
                )

                print(f"\nAssistant: {response_message}")

                if transfer_note:
                    print(f"\n{transfer_note}")

                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response_message})

                if reset_history:
                    logger.info("Resetting chat history")
                    chat_history = []

            finally:
                spinner.stop()

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
