# Context for Rinha de Backend 2026 - Fraud Detection

This document contains all the essential context, rules, and constraints required to implement the Rinha de Backend 2026 challenge. Pass this file as context to your AI assistant in future sessions.

## Challenge Goal
Build a fraud detection API for card transactions using vector search. 
For each incoming transaction:
1. Transform the payload into a 14-dimensional vector, normalized between `0.0` and `1.0`.
2. Search a reference dataset of 3 million vectors for the 5 nearest neighbors using **Euclidean distance**.
3. Compute `fraud_score = number_of_frauds_among_the_5 / 5`.
4. Decide `approved = fraud_score < 0.6`.

## Architecture & Constraints
- **Infrastructure**: Must have at least 1 Load Balancer and 2 API instances distributing load in round-robin.
- **Limits**: The sum of resource limits across all services must not exceed **1 CPU and 350 MB of memory**.
- **Network**: Docker compose using `bridge` network. `host` and `privileged` modes are forbidden.
- **Port**: Load balancer must respond on port **9999**.

## API Endpoints

### 1. `GET /ready`
Must return `HTTP 2xx` once the API is fully loaded (data is in memory) and ready to receive traffic.

### 2. `POST /fraud-score`
Receives the transaction payload and returns the decision.

**Example Request Payload:**
```json
{
  "id": "tx-123",
  "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
  "customer":    { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-001"] },
  "merchant":    { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
  "terminal":    { "is_online": false, "card_present": true, "km_from_home": 13.7 },
  "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8 }
}
```

**Response Payload:**
```json
{ "approved": false, "fraud_score": 0.8 }
```

## Dataset (Reference Files)
Located in `/resources/`. These should be loaded into memory once at startup to avoid I/O bottlenecks:
1. `references.json.gz`: 3,000,000 labeled vectors (`{"vector": [...], "label": "fraud" | "legit"}`).
2. `mcc_risk.json`: Key-value mapping of MCC codes to risk scores. Default missing keys to `0.5`.
3. `normalization.json`: Limits for normalization (e.g., `max_amount: 10000`).

## Vectorization Rules (14 Dimensions)
Apply the `clamp(value)` function to keep numeric values within `[0.0, 1.0]`.

| Index | Dimension | Formula |
|---|---|---|
| 0 | `amount` | `clamp(transaction.amount / max_amount)` |
| 1 | `installments` | `clamp(transaction.installments / max_installments)` |
| 2 | `amount_vs_avg` | `clamp((transaction.amount / customer.avg_amount) / amount_vs_avg_ratio)` |
| 3 | `hour_of_day` | `hour(transaction.requested_at) / 23` (0-23, UTC) |
| 4 | `day_of_week` | `day_of_week(transaction.requested_at) / 6` (mon=0, sun=6) |
| 5 | `minutes_since_last_tx` | `clamp(minutes / max_minutes)` or **`-1` if `last_transaction: null`** |
| 6 | `km_from_last_tx` | `clamp(last_transaction.km_from_current / max_km)` or **`-1` if `last_transaction: null`** |
| 7 | `km_from_home` | `clamp(terminal.km_from_home / max_km)` |
| 8 | `tx_count_24h` | `clamp(customer.tx_count_24h / max_tx_count_24h)` |
| 9 | `is_online` | `1` if `terminal.is_online`, else `0` |
| 10 | `card_present` | `1` if `terminal.card_present`, else `0` |
| 11 | `unknown_merchant` | `1` if `merchant.id` is NOT in `customer.known_merchants`, else `0` |
| 12 | `mcc_risk` | `mcc_risk.json[merchant.mcc]` (default `0.5`) |
| 13 | `merchant_avg_amount` | `clamp(merchant.avg_amount / max_merchant_avg_amount)` |

*Crucial Detail*: The `-1` for indices 5 and 6 when `last_transaction` is `null` is the only time values fall outside `0.0-1.0`. Do not filter these out!
