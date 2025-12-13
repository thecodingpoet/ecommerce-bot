"""
Pydantic models for agent responses and data structures.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class BaseAgentResponse(BaseModel):
    """Base response class for all agent responses."""

    message: str = Field(description="Natural language response to the user")


class SubAgentResponse(BaseAgentResponse):
    """Base response for sub-agents that can request transfers to other agents."""

    transfer_to_agent: Optional[str] = Field(
        None,
        description="Agent to transfer to: 'rag' for product search, 'order' for purchases, or None to continue with current agent",
    )


class ProductInfo(BaseModel):
    """Structured product information returned by the agent."""

    product_id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    description: str = Field(description="Product description")
    price: float = Field(description="Product price in dollars")
    category: str = Field(description="Product category")
    stock_status: str = Field(description="Stock availability status")


class RAGResponse(SubAgentResponse):
    """Structured response from the RAG agent."""

    products: List[ProductInfo] = Field(
        default_factory=list, description="List of relevant products found"
    )


class OrderResponse(SubAgentResponse):
    """Structured response from Order Agent."""

    status: Literal["collecting_info", "confirming", "completed", "failed"] = Field(
        description="Current status of order process"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of missing required fields (e.g., 'quantity', 'email')",
    )
    order_summary: Optional[dict] = Field(
        None, description="Order summary when ready to confirm"
    )
    order_id: Optional[str] = Field(
        None, description="Order ID after successful creation"
    )


class OrchestratorResponse(BaseAgentResponse):
    """Structured response from Orchestrator."""

    agent_used: str = Field(
        description="Which agent handled the request: 'rag', 'order', or 'orchestrator'."
    )
