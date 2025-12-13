"""Interactive console for the e-commerce chatbot."""

import json
from pathlib import Path

from database import OrderDatabase

# Initialize database
db = OrderDatabase("data/ecommerce.db")

# Load products for easy access
products_path = Path("data/products.json")
if products_path.exists():
    with open(products_path) as f:
        products = json.load(f)
else:
    products = []


# Helper functions
def find_product(query: str):
    """Find products by name or ID."""
    query_lower = query.lower()
    results = [
        p
        for p in products
        if query_lower in p["name"].lower() or query_lower in p["product_id"].lower()
    ]
    return results


def create_test_order():
    """Create a test order quickly."""
    items = [
        {
            "product_id": "TECH-001",
            "product_name": "MacBook Pro 16-inch",
            "quantity": 1,
            "unit_price": 2499.99,
        }
    ]
    return db.create_order("Test User", "test@example.com", items)


# Banner
banner = """
╔════════════════════════════════════════════════════════════╗
║          E-Commerce Chatbot Interactive Console            ║
╚════════════════════════════════════════════════════════════╝

Available objects:
  db         - OrderDatabase instance
  products   - List of all products from products.json
  
Useful functions:
  find_product("macbook")     - Search for products
  create_test_order()         - Create a quick test order
  
Database methods (use db.):
  create_order(name, email, items)
  get_order_by_id(order_id)
  get_all_orders()
  get_orders_by_email(email)
  get_last_order()
  update_order_status(order_id, status)
  delete_order(order_id)
  get_order_count()

Examples:
  >>> db.get_last_order()
  >>> db.get_all_orders()
  >>> find_product("macbook")
  >>> create_test_order()
  >>> db.get_order_count()

Type 'exit()' or Ctrl+D to quit.
"""

print(banner)

try:
    from IPython import embed

    embed(colors="neutral")
except ImportError:
    import code

    code.interact(local=locals(), banner="")
