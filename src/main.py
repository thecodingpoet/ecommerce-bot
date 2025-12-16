"""
E-commerce chatbot CLI and Web UI.
Interactive assistant for product search and order placement.
"""

import argparse
import logging
from typing import List

from dotenv import load_dotenv

from agents.orchestrator import Orchestrator
from utils.logger import setup_logger
from utils.spinner import Spinner
import gradio as gr

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


def run_cli(verbose: bool = False):
    """Run CLI interface - E-commerce assistant with product search and ordering.

    Args:
        verbose: Enable verbose output for debugging
    """

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
                logger.debug(f"Orchestrator state: {orchestrator._state.value}")
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


def run_web_ui(
    verbose: bool = False, server_port: int = 7860, server_name: str = "127.0.0.1"
):
    """Run web UI using Gradio - E-commerce assistant with product search and ordering.

    Args:
        verbose: Enable verbose output for debugging
        server_port: Port to run the Gradio server on
        server_name: Hostname/IP to bind the server to
    """
    logger = setup_logging(verbose)
    orchestrator = Orchestrator()
    chat_history = []

    def chat_fn(message: str, history: List[List[str]]) -> str:
        """Handle chat messages from Gradio interface.

        Args:
            message: User's message
            history: Chat history in Gradio format (ignored - we use our own)

        Returns:
            Assistant's response
        """
        if not message.strip():
            return ""

        try:
            logger.debug(f"Orchestrator state: {orchestrator._state.value}")
            logger.debug(f"Chat history length: {len(chat_history)}")

            response = orchestrator.invoke(message, chat_history=chat_history)
            response_message = response.message

            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response_message})

            return response_message

        except Exception as e:
            logger.error(f"Error: {e}")
            if verbose:
                logger.exception("Full traceback:")
            return f"❌ Error: {e}\nPlease try again."

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            show_label=False,
            autoscroll=True,
        )

        gr.ChatInterface(
            fn=chat_fn,
            title="🛍️ Your AI Shopping Assistant",
            description=(
                "Welcome! I’m here to help you shop:\n\n"
                "• 🔎 Find products you’re looking for\n"
                "• ℹ️ See details like pricing and availability\n"
                "• 🛒 Order items right here in the chat"
            ),
            chatbot=chatbot,
            textbox=gr.Textbox(
                placeholder="What are you shopping for today?",
                container=False,
                scale=7,
            ),
        )

    print(f"\n🚀 Starting web UI on http://{server_name}:{server_port}")
    print("Press Ctrl+C to stop the server\n")

    demo.launch(
        server_port=server_port,
        server_name=server_name,
        inbrowser=True,
        theme=gr.themes.Default(),
    )


def main(ui: bool = False, verbose: bool = False, port: int = 7860):
    """Main entry point - E-commerce assistant with product search and ordering.

    Args:
        ui: Launch web UI instead of CLI
        verbose: Enable verbose output for debugging
        port: Port for web UI (only used if ui=True)
    """
    if ui:
        run_web_ui(verbose=verbose, server_port=port)
    else:
        run_cli(verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E-commerce chatbot CLI and Web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          # Run CLI interface (default)
            python src/main.py

          # Run CLI with verbose logging
            python src/main.py --verbose

          # Run web UI
            python src/main.py --ui

          # Run web UI on custom port
            python src/main.py --ui --port 8080
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch web UI instead of CLI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web UI (default: 7860)",
    )
    args = parser.parse_args()

    main(ui=args.ui, verbose=args.verbose, port=args.port)
