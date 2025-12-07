"""
E-commerce chatbot main entry point.
Interactive testing for RAG, Order, and Orchestrator agents.
"""

from dotenv import load_dotenv

from agents.order_agent import OrderAgent
from agents.rag_agent import RAGAgent

load_dotenv()


def test_rag_agent():
    """Test RAG Agent with product queries."""
    agent = RAGAgent()

    queries = [
        "What laptops do you have?",
        "I'm looking for wireless headphones",
        "Do you have any gaming products?",
        "Show me products under $100",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = agent.invoke(query)

        print(f"\nANSWER:")
        print(result.answer)
        print(f"\nProducts retrieved: {len(result.products)}")


def test_order_agent():
    """Interactive chat with Order Agent."""
    agent = OrderAgent()
    chat_history = []

    print("\n" + "=" * 70)
    print("ORDER AGENT - Interactive Chat")
    print("=" * 70)
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 70)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            response = agent.invoke(user_input, chat_history=chat_history)

            print(f"\nAgent: {response.message}")

            if response.status == "completed":
                print(f"\n✅ Order completed! Order ID: {response.order_id}")

                continue_prompt = (
                    input("\nWould you like to place another order? (yes/no): ")
                    .strip()
                    .lower()
                )
                if continue_prompt != "yes":
                    print("\nThank you for using our service!")
                    break
                else:
                    chat_history = []
                    continue

            elif response.status == "failed":
                print("\n❌ Order failed. Please try again.")
                retry_prompt = (
                    input("\nWould you like to try again? (yes/no): ").strip().lower()
                )
                if retry_prompt != "yes":
                    break
                else:
                    chat_history = []
                    continue

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response.message})

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def test_handoff_pattern():
    """Interactive chat with agent handoff pattern."""
    rag_agent = RAGAgent()
    order_agent = OrderAgent()
    chat_history = []

    # Start with RAG Agent by default
    current_agent = "rag"

    print("\n" + "=" * 70)
    print("E-COMMERCE ASSISTANT - Multi-Agent Handoff Pattern")
    print("=" * 70)
    print("I can help you find products and place orders!")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 70)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            # Route to current agent
            if current_agent == "rag":
                response = rag_agent.invoke(user_input, chat_history=chat_history)

                print(f"\nAssistant: {response.answer}")

                # Check for transfer
                if response.transfer_to_agent == "order":
                    current_agent = "order"
                    print("\n[Transferring to Order Specialist...]")

                # Update chat history
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response.answer})

            elif current_agent == "order":
                response = order_agent.invoke(user_input, chat_history=chat_history)

                print(f"\nAssistant: {response.message}")

                # Show order status
                if response.status == "completed":
                    print(f"\n✅ Order completed! Order ID: {response.order_id}")
                    # Reset chat history for new order
                    chat_history = []
                    current_agent = "rag"  # Return to RAG for next interaction
                elif response.status == "failed":
                    print("\n❌ Order failed.")
                    chat_history = []
                    current_agent = "rag"

                # Check for transfer back to RAG
                if response.transfer_to_agent == "rag":
                    current_agent = "rag"
                    print("\n[Transferring to Product Search...]")

                # Update chat history
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response.message})

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def test_orchestrator():
    """Interactive chat with Orchestrator Agent."""
    from agents.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    chat_history = []

    print("\n" + "=" * 70)
    print("ORCHESTRATOR - Unified E-Commerce Assistant")
    print("=" * 70)
    print("Ask about products or place orders - I'll route you automatically!")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 70)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            response = orchestrator.invoke(user_input, chat_history=chat_history)

            print(f"\nAssistant: {response.message}")
            print(f"[Routed to: {response.agent_used}]")

            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response.message})

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def main():
    """Main entry point - choose which agent to test."""
    print("\n" + "=" * 70)
    print("E-COMMERCE CHATBOT - TEST MODE")
    print("=" * 70)
    print("1. Test RAG Agent (Product Search)")
    print("2. Test Order Agent (Place Orders)")
    print("3. Test Handoff Pattern (Multi-Agent - RECOMMENDED)")
    print("4. Test Orchestrator (Supervisor Pattern - Legacy)")
    print("=" * 70)

    choice = input("\nSelect agent to test (1/2/3/4): ").strip()

    if choice == "1":
        test_rag_agent()
    elif choice == "2":
        test_order_agent()
    elif choice == "3":
        test_handoff_pattern()
    elif choice == "4":
        test_orchestrator()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
