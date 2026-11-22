# API Key & Data Security Guidelines
- Never commit `.env` files, API tokens, or credentials to version control.
- Rotate production API keys every 90 days.
- All customer data at rest must be encrypted using AES-256, and data in transit must use TLS 1.2 or higher.
