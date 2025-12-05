"""
E-commerce chatbot main entry point.
Interactive testing for RAG and Order agents.
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


def main():
    """Main entry point - choose which agent to test."""
    print("\n" + "=" * 70)
    print("E-COMMERCE CHATBOT - TEST MODE")
    print("=" * 70)
    print("1. Test RAG Agent (Product Search)")
    print("2. Test Order Agent (Place Orders)")
    print("=" * 70)

    choice = input("\nSelect agent to test (1/2): ").strip()

    if choice == "1":
        test_rag_agent()
    elif choice == "2":
        test_order_agent()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
