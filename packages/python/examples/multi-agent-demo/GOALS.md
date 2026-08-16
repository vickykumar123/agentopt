# GOALS

## Rules
- Never mention internal refund queues

## Examples
- @router Input: "I want a refund" -> classified as `billing`
- @router Input: "I cannot log in" -> classified as `technical`
- @router Input: "Please delete my account" -> classified as `account`
- @billing Input: "I want a refund" -> mentions 30-day money-back
- @billing Input: "I want a refund" -> calls the lookup_order tool
- @billing Input: "I want a refund" -> routes as tools
- @billing Input: "What does Pro cost?" -> mentions $19.99
- @billing Input: "What does Pro cost?" -> does not call the lookup_order tool
- @technical Input: "I forgot my password" -> mentions password reset link
- @technical Input: "I forgot my password" -> routes as direct
- Input: "Please delete my account" -> calls the escalate_to_human tool

## Priorities
accuracy > cost > speed
