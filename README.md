# E-Commerce AI Chatbot

An intelligent AI chatbot for e-commerce that handles product inquiries and order processing through natural conversation. The system uses RAG (Retrieval-Augmented Generation) to answer product questions and employs conversational AI to guide customers through the entire purchase journey - from product discovery to order completion - using only natural language dialogue.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)

## Features

- **🤖 Conversational AI**: Natural language interface for product search and ordering
- **🌐 Web UI**: Modern Gradio-based web interface with auto-scrolling chat window
- **💻 CLI Interface**: Command-line interface for terminal-based interactions
- **🔍 Intelligent Product Search**: RAG-powered semantic search across product catalog using ChromaDB
- **🛒 Seamless Order Processing**: Conversational checkout that collects customer details naturally through dialogue
- **🧠 Multi-Agent Architecture**: Specialized agents (Orchestrator, RAG, Order) working in concert
- **💬 Stateful Conversations**: Maintains context across multiple turns for coherent interactions
- **📦 Order Management**: Complete order lifecycle from placement to database persistence
- **🎯 Smart Intent Detection**: Automatically routes queries to appropriate specialist agents
- **🔄 Dynamic Mode Switching**: Seamlessly transitions between product search and order modes
- **💾 Dual Storage System**: 
  - Vector embeddings in ChromaDB for semantic search
  - Relational data in SQLite for orders and transactions
- **🛠️ Admin Console**: Interactive development shell for data inspection and maintenance

## Architecture

The application follows a modular multi-agent architecture powered by LangChain. It uses an Orchestrator to route queries to specialized RAG and Order agents.

For a detailed deep-dive into the system design, components, and data flow, please see the [Architecture Documentation](docs/architecture.md).

## Setup

### Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.12 or higher**
- **[uv](https://github.com/astral-sh/uv)** (recommended) or `pip`
- **OpenAI API Key** - [Get one here](https://platform.openai.com/api-keys)

### Installation

Follow these steps to set up the application:

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd ecommerce-bot
```

#### 2. Install Dependencies

Using `uv`:

```bash
uv sync
```

#### 3. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API Key:

```
OPENAI_API_KEY=sk-...
```

#### 4. Initialize Data

Populate the vector store with the initial product catalog:

```bash
uv run src/initialize_vector_store.py
```

## Usage

### Running the Chat Interface

The application supports both CLI and Web UI interfaces:

#### Command-Line Interface (CLI)
<img width="892" height="629" alt="Screenshot 2025-12-16 at 08 12 08" src="https://github.com/user-attachments/assets/a55c40e3-f498-44af-a3b6-86708052d170" />


To start the CLI conversational assistant:

```bash
uv run src/main.py
```

#### Web UI

<img width="1642" height="870" alt="Screenshot 2025-12-16 at 08 05 49" src="https://github.com/user-attachments/assets/4a5ed3ce-3a12-4b4f-b219-2729d352722c" />

To launch the web-based interface:

```bash
uv run src/main.py --ui
```

The web UI will:
- Automatically open in your default browser
- Run on `http://127.0.0.1:7860` by default
- Provide a modern chat interface with auto-scrolling

**Custom Port:**

Run the web UI on a custom port:

```bash
uv run src/main.py --ui --port 8080
```

#### Verbose Mode

Enable verbose mode for debugging and detailed logging (works with both CLI and UI):

```bash
# CLI with verbose
uv run src/main.py --verbose

# Web UI with verbose
uv run src/main.py --ui --verbose
# or use short form
uv run src/main.py --ui -v
```

Verbose mode provides:
- DEBUG level logging for all components
- Order mode state tracking
- Chat history length monitoring
- Full error tracebacks
- Agent decision visibility

#### Command-Line Options

```bash
# Show help
uv run src/main.py --help

# CLI options
uv run src/main.py                    # Run CLI (default)
uv run src/main.py --verbose          # CLI with verbose logging
uv run src/main.py -v                 # Short form for verbose

# Web UI options
uv run src/main.py --ui               # Run web UI on default port (7860)
uv run src/main.py --ui --port 8080   # Run web UI on custom port
uv run src/main.py --ui --verbose     # Web UI with verbose logging
```

### Developer Console

<img width="947" height="829" alt="Screenshot 2025-12-16 at 08 16 46" src="https://github.com/user-attachments/assets/94289124-4a7c-4b49-873c-1908278d202e" />


To access the interactive developer console for debugging and testing:

```bash
uv run src/console.py
```

This opens an interactive Python shell with pre-initialized objects:

- **`products`**: Instance of `ProductCatalog` for managing product data.
  - Usage: `products.get_product("TECH-001")`, `products.get_all_products()`
- **`orders`**: Instance of `OrderDatabase` for managing order records.
  - Usage: `orders.get_last_order()`, `orders.get_order_count()`

Use this console for deep inspection of the database, manual testing of data retrieval, and debugging order states.

## Testing

For manual testing scenarios and expected conversation flows, see the [Conversation Test Guide](examples/test_conversations.md).
This guide covers:
- Product price queries
- Multi-turn discussions
- Order confirmation flows
- Ambiguous query handling
- Edge cases and error handling

## Limitations

- **Single User Web UI**: The current Web UI implementation shares state (conversation history and shopping cart) across all connected users. It is designed for local, single-user testing only. For multi-user deployments, session management logic would need to be added.
- **In-Memory Shopping Cart**: Shopping cart data is stored in memory and will be lost if the application is restarted.
- **No Payment Processing**: The application simulates the checkout process and does not integrate with real payment gateways or shipping providers.
