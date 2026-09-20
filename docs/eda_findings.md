# Phase 2 — Analytics & Exploratory Data Analysis (EDA) Findings

This document summarizes the 5 key analytical findings discovered during the exploration of the cleaned 7,999-row English ticket warehouse (`data/warehouse.db`).

---

### Finding 1: Heavy Imbalance in Ticket Volumes by Queue
The distribution of tickets across support queues follows a steep power-law curve:
- **Technical** is by far the largest queue, accounting for over **2,168 tickets (27.1%)**.
- **Security** is second with **1,351 tickets (16.9%)**.
- **Bug** is third with **791 tickets (9.9%)**.
- Combined, the top 5 queues (`Technical`, `Security`, `Bug`, `Feedback`, `Feature`) represent more than **69% of all ticket traffic**, while specialized queues (such as `Malware`, `Disruption`, `Device`) have fewer than 50 cases each.

### Finding 2: Priority Distribution Skew and Resolution Urgency
Across all 7,999 English tickets:
- **Medium priority** represents the vast majority (~58%).
- **Low priority** represents approximately 22%.
- **High priority** accounts for approximately 20% of tickets.
- Specific queues show a disproportionate surge in high-priority tickets:
  - `Security` and `Breach` tickets have the highest ratio of high-priority tickets (>38% marked high).
  - Conversely, `Feedback` and `Documentation` tickets are overwhelmingly low or medium priority (<5% high priority).

### Finding 3: Queue Answer Length vs Complexity
Analyzing the average character length of agent resolutions across queues reveals significant variance:
- **Crash and Multi-tag Infrastructure issues** require the longest answers, averaging **829 characters** per resolution.
- **Device & Security configuration** solutions average **806 and 660 characters**, requiring step-by-step diagnostic workflows.
- Routine queues such as **Inquiry** and **General** require the shortest answers (averaging **under 250 characters**), indicating straightforward templated responses.

### Finding 4: Top Terms and Keyword Disambiguation
Term-frequency analysis (TF-IDF) across ticket bodies reveals distinct semantic signatures per queue:
- `Billing`: "invoice", "charge", "refund", "subscription", "credit card", "renewal", "payment".
- `Security`: "password", "reset", "unauthorized", "login", "credentials", "access", "authentication".
- `Technical`: "crash", "error code", "connection", "vpn", "server", "timeout", "latency".
The distinct vocabulary confirms that automated classification via NLP and dense vector embedding for RAG will have strong discriminative power.

### Finding 5: High Redundancy in Common IT Problems
A significant portion of incoming tickets represent recurring, solved problems:
- VPN dropouts and authentication issues after credential changes appear repeatedly across multiple user accounts.
- Double-charge refund requests follow identical workflows (request account ID -> verify transaction date -> issue refund).
This high degree of redundancy validates the core objective of the AI support assistant: retrieving and citing proven historical resolutions saves substantial human support hours.
