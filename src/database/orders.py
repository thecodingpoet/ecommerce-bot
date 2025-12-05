"""Database module for order management using SQLAlchemy ORM."""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

Base = declarative_base()


class Order(Base):
    """Order table schema."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String, unique=True, nullable=False, index=True)
    customer_name = Column(String, nullable=False)
    customer_email = Column(String, nullable=False)
    total_amount = Column(Float, nullable=False)
    status = Column(String, default="pending", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    items = relationship(
        "OrderItem", back_populates="order", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Order(order_id='{self.order_id}', customer='{self.customer_name}', total={self.total_amount})>"


class OrderItem(Base):
    """Order items table schema."""

    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String, ForeignKey("orders.order_id"), nullable=False)
    product_id = Column(String, nullable=False)
    product_name = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    subtotal = Column(Float, nullable=False)

    order = relationship("Order", back_populates="items")

    def __repr__(self):
        return f"<OrderItem(product='{self.product_name}', qty={self.quantity}, subtotal={self.subtotal})>"


class OrderDatabase:
    """Database manager for order operations."""

    def __init__(self, db_path: str = "data/ecommerce.db"):
        """Initialize database connection and create tables.

        Args:
            db_path: Path to SQLite database file
        """
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _get_session(self) -> Session:
        """Create a new database session."""
        return self.SessionLocal()

    def create_order(
        self,
        customer_name: str,
        customer_email: str,
        items: List[dict],
    ) -> Order:
        """Create a new order with items.

        Args:
            customer_name: Customer's full name
            customer_email: Customer's email address
            items: List of dicts with keys: product_id, product_name, quantity, unit_price

        Returns:
            Created Order object

        Example:
            >>> items = [
            ...     {
            ...         "product_id": "TECH-001",
            ...         "product_name": "MacBook Pro",
            ...         "quantity": 2,
            ...         "unit_price": 2499.99
            ...     }
            ... ]
            >>> order = db.create_order("John Doe", "john@example.com", items)
        """
        session = self._get_session()
        try:
            order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
            total_amount = sum(item["quantity"] * item["unit_price"] for item in items)

            order = Order(
                order_id=order_id,
                customer_name=customer_name,
                customer_email=customer_email,
                total_amount=total_amount,
                status="pending",
            )

            for item in items:
                order_item = OrderItem(
                    order_id=order_id,
                    product_id=item["product_id"],
                    product_name=item["product_name"],
                    quantity=item["quantity"],
                    unit_price=item["unit_price"],
                    subtotal=item["quantity"] * item["unit_price"],
                )
                order.items.append(order_item)

            session.add(order)
            session.commit()
            session.refresh(order)

            # Eagerly load items before closing session
            _ = order.items

            return order

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by its order_id.

        Args:
            order_id: Unique order identifier

        Returns:
            Order object if found, None otherwise
        """
        session = self._get_session()
        try:
            order = session.query(Order).filter(Order.order_id == order_id).first()
            if order:
                _ = order.items
            return order
        finally:
            session.close()

    def get_all_orders(self, limit: int = 100) -> List[Order]:
        """Retrieve all orders, most recent first.

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of Order objects
        """
        session = self._get_session()
        try:
            orders = (
                session.query(Order)
                .order_by(Order.created_at.desc())
                .limit(limit)
                .all()
            )
            for order in orders:
                _ = order.items
            return orders
        finally:
            session.close()

    def get_orders_by_email(self, email: str) -> List[Order]:
        """Retrieve all orders for a specific customer email.

        Args:
            email: Customer email address

        Returns:
            List of Order objects
        """
        session = self._get_session()
        try:
            orders = (
                session.query(Order)
                .filter(Order.customer_email == email)
                .order_by(Order.created_at.desc())
                .all()
            )
            for order in orders:
                _ = order.items
            return orders
        finally:
            session.close()

    def update_order_status(self, order_id: str, status: str) -> Optional[Order]:
        """Update the status of an order.

        Args:
            order_id: Unique order identifier
            status: New status (e.g., 'pending', 'confirmed', 'shipped', 'delivered', 'cancelled')

        Returns:
            Updated Order object if found, None otherwise
        """
        session = self._get_session()
        try:
            order = session.query(Order).filter(Order.order_id == order_id).first()
            if order:
                order.status = status
                order.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(order)
                _ = order.items
            return order
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_order(self, order_id: str) -> bool:
        """Delete an order and its items (cascade).

        Args:
            order_id: Unique order identifier

        Returns:
            True if deleted, False if not found
        """
        session = self._get_session()
        try:
            order = session.query(Order).filter(Order.order_id == order_id).first()
            if order:
                session.delete(order)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_order_count(self) -> int:
        """Get total number of orders in database.

        Returns:
            Count of orders
        """
        session = self._get_session()
        try:
            return session.query(Order).count()
        finally:
            session.close()
