# Vireo Audio — Support Intelligence & Triage AI

A self-contained, evidence-based AI support ticket classification and queue analytics system developed for **Vireo Audio Pvt. Ltd.**

This tool analyzes support ticket text (`customer_message` and `agent_notes`), identifies intake chatbot routing discrepancies, calculates model-predicted monthly ticket volumes by category and team, estimates internal transfer process-cost exposure using policy benchmarks (₹305/transfer), and provides structured decision support for support capacity planning.

**Primary Analysis Period:** January 1, 2025 through June 30, 2026 inclusive (11,641 support tickets analyzed across live Helpdesk and migrated Freshdesk systems).

---

## 🚀 Quickstart Guide (Clean Machine Setup)

### 1. Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.13, 3.14 fully supported)
- No external API keys, database servers, or cloud credentials required (100% offline)

### 2. Installation

Clone or extract the repository, then navigate to the folder:
```bash
cd vireo-support-ai
```

Create and activate a clean virtual environment:
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run Validation & Evaluation (CLI)
To run the automated evaluation suite and inspect validation audit metrics in the terminal:
```bash
python evaluate.py
```

### 4. Launch the Interactive Dashboard (Streamlit UI)
To launch the interactive web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧠 How Classification & Triage Work

### 1. Classification Engine (`classifier.py`)
The pipeline uses a two-tier hybrid architecture:
1. **Domain-Specific Pattern Rules:** High-precision regex matching for established support intents (e.g. tracking updates, payment gateway failures, OTP logins, RMA warranty claims, battery drain).
2. **Statistical ML Fallback:** A TF-IDF vectorizer (unigrams + bigrams) paired with a class-balanced Logistic Regression model to classify unstructured text when explicit keywords are absent.

Each classified ticket generates:
- `ticket_id`: Unique identifier
- `original_category`: Initial category tag assigned by intake chatbot
- `predicted_category`: AI-predicted support category (1 of 11 valid categories)
- `predicted_team`: Responsible team based on category and channel routing rules
- `confidence`: Probabilistic confidence score (0.0 to 1.0)
- `short_reason`: Human-interpretable rationale for the prediction

### 2. Team Routing Rules (Support Policy §6)
- **Delivery & Shipping** $\rightarrow$ `Logistics`
- **Billing & Payments** $\rightarrow$ `Billing`
- **Returns & Refunds** $\rightarrow$ `Returns Desk`
- **Warranty & Repair** $\rightarrow$ `Escalations & Warranty` (Tier 2 Certified)
- **Tier 1 Frontline** (*Product Enquiry, Connectivity, Battery, Audio, App, Account, Other*):
  - `chat` / `social` $\rightarrow$ `Chat Frontline`
  - `email` $\rightarrow$ `Email Frontline`
  - `voice` $\rightarrow$ `Voice Frontline`

---

## 🔍 How Validation Works

### 1. Discarding the Circular 95% Heuristic Metric
During development, an earlier iteration reported a "95.0% auditor agreement" on a 100-ticket sample. Upon rigorous audit of the benchmark script (`scratch/generate_validated_sample.py`), it was discovered that the "auditor" labels had been programmatically initialized with the AI model's own predictions (`val_cat = pred`) and then modified with simple regex heuristics.

**That 95% metric was circular, artificial, and methodologically invalid. It has been permanently discarded.**

### 2. The Transparent Human-Review Protocol (`manual_validation_reviews.csv`)
For V3, a genuine human-in-the-loop validation system was implemented:
- **Reproducible 30-Ticket Sample:** Exactly 30 tickets were sampled from the primary analysis period (Jan 1, 2025 – Jun 30, 2026) using fixed seed `random_state=42`.
- **Blinded Evaluation UI:** Reviewers audit tickets via the interactive Streamlit dashboard (**Tab 3: Manual Validation Audit**). The UI displays only `ticket_id`, `channel`, `customer_message`, `agent_notes`, and `original_category`.
- **Strict Anti-Bias Controls:** The AI predicted category and team are **strictly blinded** until after the human validator selects the category, selects the team, and submits the review. The input fields are never prefilled with AI predictions or heuristic overrides.
- **Post-Submission Revelation:** Once saved, the UI reveals the model's prediction, records match/mismatch status, and persists the review to `manual_validation_reviews.csv`.
- **Conditional Metric Calculation:** Agreement rates and error analyses are calculated strictly over completed human reviews.

### 3. Final Human-Validation Audit Results (30-Ticket Sample)
All 30 tickets in the fixed audit sample were independently reviewed by a human reviewer without previewing AI predictions. The completed audit revealed:

- **Human Reviewer vs. AI Category Agreement:** **30 / 30 tickets (100.0%)**
- **Human Reviewer vs. AI Responsible Team Agreement:** **29 / 30 tickets (96.7%)**
- **Original Intake Category vs. Human Reviewer:** **21 / 30 tickets (70.0%)** (indicating that 30% of original intake tags in this sample were miscategorized by the initial bot)
- **Observed Team Routing Discrepancy (1 ticket / 30):**
  - **Ticket `TK-247547`:** Customer reported audio crackling/distortion on their headphones while also inquiring about warranty coverage. The AI model routed to `Chat Frontline` based on primary technical troubleshooting keywords, whereas human review escalated directly to `Escalations & Warranty` (Tier 2 Certified RMA).

### 4. Important Methodological Distinction: Sample Agreement vs. Production Accuracy
- **Sample Agreement Only:** The 100% category and 96.7% team figures reflect agreement on a **controlled, blinded 30-ticket sample**, **NOT guaranteed production accuracy** across the entire 11,641-ticket dataset.
- **Directional Scope:** The 30-ticket sample is directional and not sufficient to estimate production-wide accuracy across the entire support workload. It serves as a qualitative check on model behavior and edge-case routing rather than an exhaustive statistical benchmark.

---

## 📊 Summary of Observed Workload & Queue Differences (Jan 2025 – Jun 2026)

Comparing initial intake bot routing against model-predicted workloads across all 11,641 primary analysis period tickets:

| Support Team | Initial Bot Assigned Volume | Model-Predicted Workload | Volume Variance | Model Workload Share | Operational Tier Level |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistics** | 1,905 | **3,334** | **+1,429 (+75.0%)** | **28.6%** | Tier 1 / Specialized |
| **Chat Frontline** | 3,030 | 2,308 | -722 | 19.8% | Tier 1 |
| **Returns Desk** | 1,049 | 1,758 | +709 (+67.6%) | 15.1% | Tier 1 / Specialized |
| **Billing** | 2,425 | **1,601** | **-824 (-34.0%)** | **13.8%** | Tier 1 / Specialized |
| **Email Frontline** | 1,807 | 1,374 | -433 | 11.8% | Tier 1 |
| **Voice Frontline** | 900 | 710 | -190 | 6.1% | Tier 1 |
| **Escalations & Warranty** | 525 | 556 | +31 | 4.8% | **Tier 2 (Certified RMA)** |

### Analytical Insights for Leadership
1. **The Billing Queue Distortion:** 46.3% of tickets initially assigned to Billing were shipment delivery delays or return inquiries (caused by customers mentioning "paid", "money debited", or "invoice" in shipping contexts). True Billing workload is model-estimated at 13.8%.
2. **Logistics Queue Demand:** Logistics represents ~28.6% of estimated workload, corroborating operational observations that Logistics is absorbing heavy hand-offs and experiencing longer resolution times.
3. **Tier 2 Policy Nuance:** Support Policy §6 states Tier 2 cases are certified, multi-touch RMA investigations measured in resolution days, not weekly volume. Capacity planning must weigh high-volume Tier 1 handling against Tier 2 technical complexity.

---

## 💰 Transfer Process-Cost Exposure

- **Observed Team Routing Discrepancies:** Across the 11,641 tickets in the analysis period, the model identified **3,020 tickets (25.9%)** where the initial chatbot assigned the ticket to a different team than the model-predicted responsible queue.
- **Policy Cost Benchmark:** Support Operating Policy v3.2 §4 defines the internal transfer standard at **₹305 per hand-off** (re-handling and administrative overhead).
- **Estimated Process-Cost Exposure:** Multiplying the 3,020 misrouted tickets by the ₹305 transfer-equivalent yields an estimated **₹921,100 (₹9.21 Lakh)** in process-cost exposure.
- **Operational Nature of the Metric:** This figure reflects **potential process friction, re-handling waste, and operational drag** across queues. It is **NOT guaranteed cash savings or direct monetary recovery**, as actual savings depend on whether tickets were actively re-transferred in practice, agent workflows, and the implementation of automated intake routing fixes.

---

## ⚠️ Limitations & Known Issues

- **Probabilistic Predictions:** AI outputs are statistical estimates derived from text patterns, not infallible ground truth.
- **Validation Sample Scope:** Audit results reflect a 30-ticket human-reviewed sample designed for rapid qualitative verification; they do not represent full-population verification.
- **Multi-Intent Ambiguity:** When customers combine complaints (e.g. reporting a defective product while demanding an immediate return/refund), single-label categorization may capture the defect symptom rather than the primary resolution intent.
- **Legacy System Data Missingness:** Per Support Policy §9, tickets migrated from Freshdesk (prior to 14 Sep 2025) lack the `transfers` field (stored as blank/NaN, not 0).
- **Batch Processing Scope:** The system analyzes offline static CSV exports; real-time helpdesk API webhooks and automated routing integrations are reserved for future phases.

---

## 📁 Repository Structure

```text
vireo-support-ai/
├── app.py                         # Streamlit dashboard (6 tabs including Manual Validation Audit)
├── classifier.py                  # Hybrid rule + ML ticket classifier
├── analysis.py                    # Volume metrics, misrouting analysis, and transfer costing
├── evaluate.py                    # Standalone evaluation & manual validation audit script
├── manual_validation_reviews.csv  # 30-ticket reproducible sample for genuine human validation (Seed=42)
├── validation_sample_100.csv      # Historical archive (discarded circular 95% metric evidence)
├── requirements.txt               # Python dependencies (pandas, scikit-learn, streamlit, plotly, altair)
├── .gitignore                     # Standard git ignore definitions
├── README.md                      # Comprehensive setup and analytical documentation
└── PROMPTS.md                     # Complete prompt history and iteration audit trail
```
