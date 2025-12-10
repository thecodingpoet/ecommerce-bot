# E-Commerce AI Chatbot

An intelligent AI chatbot for e-commerce that handles product inquiries and order processing through natural conversation. The system uses RAG (Retrieval-Augmented Generation) to answer product questions and employs conversational AI to guide customers through the entire purchase journey - from product discovery to order completion - using only natural language dialogue.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)

## Features

- **🤖 Conversational AI**: Natural language interface for product search and ordering
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

To start the main conversational assistant:

```bash
uv run src/main.py
```

### Developer Console

To access the interactive developer console for debugging and testing:

```bash
uv run src/console.py
```

This opens a Python shell with pre-initialized `db` (database connection) and `products` variables, plus helpful functions like `find_product()` and `create_test_order()`. Use this for data inspection, manual testing, and debugging.
