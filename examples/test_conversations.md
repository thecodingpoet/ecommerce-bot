# Test Conversations

This document contains manual test cases for validating the e-commerce chatbot's conversation flows. Each test case includes the conversation transcript and expected outcomes.

---

## Table of Contents

1. [Product Price Query](#1-product-price-query)
2. [Multi-turn Product Discussion](#2-multi-turn-product-discussion)
3. [Order Confirmation with Extraction](#3-order-confirmation-with-extraction)
4. [Ambiguous Query Handling](#4-ambiguous-query-handling)
5. [Invalid Order Rejection](#5-invalid-order-rejection)

---

## 1. Product Price Query

**Objective:** Verify the RAG agent correctly retrieves and displays product pricing information.

### Test Case 1.1: Single Product Price Query

**User Input:**
```
How much does the MacBook Pro 16-inch cost?
```

**Expected Outcome:**
- Agent uses RAG agent to search for the product
- Response includes:
  - Product name: "MacBook Pro 16-inch"
  - Price: $2,499.99
  - Stock status: in_stock

**Verification Criteria:**
- ✅ Correct product identified
- ✅ Accurate price returned
- ✅ No hallucinated information

---

### Test Case 1.2: Product Query by ID

**User Input:**
```
What's the price of TECH-007?
```

**Expected Outcome:**
- Agent identifies product: Dell XPS 13 Laptop
- Returns price: $1,299.99
- Stock status: in_stock

**Verification Criteria:**
- ✅ Product ID correctly resolved to product name
- ✅ Accurate pricing information

---

### Test Case 1.3: Category-Based Price Query

**User Input:**
```
What headphones do you have and how much are they?
```

**Expected Outcome:**
- Returns multiple headphone options including:
  - AirPods Pro 2nd Gen ($249.99)
  - Sony WH-1000XM5 Headphones ($399.99)
- Lists prices for each headphone product

**Verification Criteria:**
- ✅ Multiple products returned
- ✅ All prices are accurate
- ✅ Products are from relevant category

---

## 2. Multi-turn Product Discussion

**Objective:** Verify the system maintains context across multiple conversation turns.

### Test Case 2.1: Progressive Product Exploration

**Conversation Flow:**

| Turn | User | Expected Assistant Response |
|------|------|----------------------------|
| 1 | "I'm looking for a laptop" | Lists available laptops (MacBook Pro, Dell XPS 13) with brief descriptions |
| 2 | "Tell me more about the Dell one" | Provides detailed info about Dell XPS 13: Intel Core i7, 16GB RAM, 512GB SSD, price $1,299.99 |
| 3 | "How does it compare to the MacBook?" | Compares Dell XPS 13 vs MacBook Pro 16-inch, highlighting specs and price differences |
| 4 | "I think I'll go with the Dell" | Recognizes purchase intent and either confirms or transfers to Order Agent |

**Verification Criteria:**
- ✅ Context maintained across turns
- ✅ "Dell one" correctly references Dell XPS 13 from turn 1
- ✅ Comparison is accurate and helpful
- ✅ Purchase intent correctly triggers order flow

---

### Test Case 2.2: Category Exploration with Follow-ups

**Conversation Flow:**

| Turn | User | Expected Assistant Response |
|------|------|----------------------------|
| 1 | "What gaming consoles do you have?" | Lists PS5, Nintendo Switch OLED, Xbox Series X |
| 2 | "Which one has good availability?" | Mentions Nintendo Switch OLED (in_stock), notes PS5 and Xbox are low_stock |
| 3 | "What's the Switch price?" | Returns $349.99 for Nintendo Switch OLED |
| 4 | "And the Xbox?" | Returns $499.99 for Xbox Series X, may note low_stock |

**Verification Criteria:**
- ✅ All gaming consoles correctly identified
- ✅ Stock status accurately reported
- ✅ Follow-up questions resolved using conversation context

---

## 3. Order Confirmation with Extraction

**Objective:** Verify the Order Agent correctly collects customer information and processes orders.

### Test Case 3.1: Complete Order Flow

**Conversation Flow:**

| Turn | User | Expected Assistant Response |
|------|------|----------------------------|
| 1 | "I want to buy the Sony WH-1000XM5 headphones" | Acknowledges order intent, adds to cart, asks for customer details |
| 2 | "My name is John Smith" | Confirms name, asks for remaining info (email, address) |
| 3 | "john.smith@email.com" | Confirms email captured, asks for shipping address |
| 4 | "123 Main Street, New York, NY 10001" | Shows order summary for confirmation |
| 5 | "Yes, confirm the order" | Creates order, provides order ID (format: ORD-{timestamp}) |

**Expected Extraction:**
```json
{
  "customer_name": "John Smith",
  "email": "john.smith@email.com",
  "shipping_address": "123 Main Street, New York, NY 10001",
  "order_items": [
    {
      "product_id": "TECH-005",
      "name": "Sony WH-1000XM5 Headphones",
      "quantity": 1,
      "price": 399.99
    }
  ],
  "total": 399.99
}
```

**Verification Criteria:**
- ✅ All customer fields correctly extracted
- ✅ Order summary accurate before confirmation
- ✅ Order ID generated and returned
- ✅ Order status transitions: collecting_info → confirming → completed

---

### Test Case 3.2: Multi-Item Order with Quantity

**User Input (single message with all info):**
```
I'd like to order 2 Yoga Mats. My name is Alice Brown, email alice@test.com, ship to 456 Oak Ave, Chicago, IL 60601
```

**Expected Outcome:**
- Extracts all information in single turn
- Order summary shows:
  - Product: Yoga Mat Premium (SPORT-003)
  - Quantity: 2
  - Unit Price: $39.99
  - Total: $79.98
- Asks for confirmation before creating order

**Verification Criteria:**
- ✅ Quantity correctly parsed as 2
- ✅ Total calculation correct (2 × $39.99)
- ✅ All customer info extracted from single message
- ✅ Confirmation requested before order creation

---

### Test Case 3.3: Order with Missing Information Prompts

**Conversation Flow:**

| Turn | User | Expected Assistant Response |
|------|------|----------------------------|
| 1 | "Buy TECH-003" | Adds AirPods Pro to cart, asks for name |
| 2 | "Sarah" | Captures name, asks for email |
| 3 | "sarah@gmail.com" | Captures email, asks for shipping address |
| 4 | "Just ship it to my house" | Politely asks for complete shipping address |
| 5 | "789 Elm St, Austin, TX 78701" | Shows order summary, asks for confirmation |

**Verification Criteria:**
- ✅ Order Agent collects info incrementally
- ✅ Handles incomplete responses gracefully
- ✅ All required fields collected before order creation

---

## 4. Ambiguous Query Handling

**Objective:** Verify the system handles vague or ambiguous queries appropriately.

### Test Case 4.1: Ambiguous Product Reference

**User Input:**
```
I want the wireless one
```

**Expected Outcome:**
- Agent asks clarifying question about which wireless product
- May list wireless products: AirPods Pro, Sony headphones, wireless mice, etc.

**Verification Criteria:**
- ✅ Does not assume which product
- ✅ Asks for clarification
- ✅ Provides helpful context or options

---

### Test Case 4.2: Intent Ambiguity

**User Input:**
```
I want a laptop
```

**Expected Outcome:**
- Could be interpreted as:
  - Browse laptops (RAG agent)
  - Purchase a laptop (Order agent)
- Agent should clarify intent or default to showing available laptops

**Verification Criteria:**
- ✅ Handles ambiguity gracefully
- ✅ Either asks for clarification OR defaults to product search
- ✅ Does not assume purchase intent immediately

---

### Test Case 4.3: Vague Quantity Request

**Conversation Flow:**

| Turn | User | Expected Assistant Response |
|------|------|----------------------------|
| 1 | "Add some AirPods to my cart" | Should ask "How many would you like?" or default to 1 with confirmation |
| 2 | "A few" | Should ask for specific quantity or confirm reasonable default |

**Verification Criteria:**
- ✅ Handles vague quantity gracefully
- ✅ Clarifies before proceeding with order

---

### Test Case 4.4: Product with Multiple Similar Results

**User Input:**
```
Do you have any Apple products?
```

**Expected Outcome:**
- Returns multiple Apple products:
  - MacBook Pro 16-inch
  - iPhone 15 Pro
  - AirPods Pro 2nd Gen
  - iPad Air 11-inch
  - Apple Watch Series 9
- Lists all with prices and availability

**Verification Criteria:**
- ✅ Semantic search returns all relevant Apple products
- ✅ Results are from actual catalog (no hallucinations)

---

## 5. Invalid Order Rejection

**Objective:** Verify the system properly handles invalid order scenarios.

### Test Case 5.1: Out-of-Stock Product Order

**User Input:**
```
I want to buy the iPhone 15 Pro
```

**Expected Outcome:**
- System detects product TECH-002 is out_of_stock
- Informs user product is unavailable
- May suggest alternatives (other phones or similar products)

**Verification Criteria:**
- ✅ Order rejected due to stock status
- ✅ Clear message about unavailability
- ✅ Does not proceed with order creation

---

### Test Case 5.2: Non-Existent Product Order

**User Input:**
```
I want to order TECH-999
```

**Expected Outcome:**
- System cannot find product with ID TECH-999
- Returns helpful error message
- May ask user to search for product first

**Verification Criteria:**
- ✅ Invalid product ID rejected
- ✅ Clear error message provided
- ✅ No order created

---

### Test Case 5.3: Invalid Email Format

**Conversation Flow:**

| Turn | User | Expected Assistant Response |
|------|------|----------------------------|
| 1 | "Buy SPORT-003" | Adds Yoga Mat to cart, asks for customer info |
| 2 | "Name: Test User, Email: not-an-email" | Should detect invalid email format and ask for valid email |

**Verification Criteria:**
- ✅ Invalid email format detected
- ✅ User prompted to provide valid email
- ✅ Order not created with invalid data

---

### Test Case 5.4: Quantity Exceeds Stock

**User Input:**
```
I want to order 1000 TECH-009 Logitech mice
```

**Expected Outcome:**
- System either:
  - Rejects order due to insufficient stock
  - Warns user about potential availability issues
- Should not blindly accept unreasonable quantity

**Verification Criteria:**
- ✅ Large quantity handled appropriately
- ✅ User informed of any limitations

---

### Test Case 5.5: Cancel Order Mid-Process

**Conversation Flow:**

| Turn | User | Expected Assistant Response |
|------|------|----------------------------|
| 1 | "I want to buy the Kindle" | Starts order process for Kindle Paperwhite |
| 2 | "Actually, never mind" | Should acknowledge cancellation and clear cart |
| 3 | "Show me some laptops instead" | Should transfer back to RAG agent for product search |

**Expected State Transitions:**
- Turn 1: INTENT → CHECKOUT (transfer to Order agent)
- Turn 2: Order Agent handles cancellation
- Turn 3: CHECKOUT → INTENT (transfer to RAG agent)

**Verification Criteria:**
- ✅ Cancellation acknowledged
- ✅ Cart cleared appropriately
- ✅ Smooth transition back to product search

---

## Test Execution Notes

### Running Tests Manually

1. Start the application:
   ```bash
   python src/main.py        # CLI mode
   python src/main.py --ui   # Web UI mode
   ```

2. Execute each conversation flow sequentially
3. Record actual responses
4. Compare against expected outcomes

### Common Issues to Watch For

- **Hallucinations**: Agent inventing products or prices not in catalog
- **Context Loss**: Agent forgetting previous conversation turns
- **State Stuck**: Orchestrator stuck in CHECKOUT mode after cancellation
- **Tool Errors**: Failures in `retrieve_products`, `add_to_cart`, or `create_order` tools

### Test Environment Requirements

- Valid `.env` file with `OPENAI_API_KEY`
- Products database populated (`data/products.json` and ChromaDB)
- Orders database accessible (`data/ecommerce.db`)

---

## Quick Reference: Product IDs for Testing

| Product ID | Name | Price | Stock Status |
|------------|------|-------|--------------|
| TECH-001 | MacBook Pro 16-inch | $2,499.99 | in_stock |
| TECH-002 | iPhone 15 Pro | $1,199.99 | out_of_stock |
| TECH-003 | AirPods Pro 2nd Gen | $249.99 | in_stock |
| TECH-005 | Sony WH-1000XM5 Headphones | $399.99 | in_stock |
| TECH-007 | Dell XPS 13 Laptop | $1,299.99 | in_stock |
| TECH-009 | Logitech MX Master 3S Mouse | $99.99 | in_stock |
| HOME-002 | Ninja Air Fryer | $129.99 | in_stock |
| SPORT-003 | Yoga Mat Premium | $39.99 | in_stock |
| BOOK-001 | Kindle Paperwhite | $149.99 | in_stock |
| GAME-001 | Nintendo Switch OLED | $349.99 | in_stock |
