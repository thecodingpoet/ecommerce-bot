"""
E-commerce chatbot main entry point.
"""

from dotenv import load_dotenv

from agents.rag_agent import RAGAgent

load_dotenv()


def main():
    """Run the e-commerce chatbot."""
    agent = RAGAgent()

    # Example queries
    queries = [
        "What laptops do you have?",
        # "I'm looking for wireless headphones",
        # "Do you have any gaming products?",
        # "Show me products under $100",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = agent.invoke(query)

        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(result["answer"])

        print(f"\nProducts retrieved: {len(result.get('products', []))}")


if __name__ == "__main__":
    main()
