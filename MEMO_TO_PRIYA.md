# MEMORANDUM

**TO:** Priya Raman  
**FROM:** Support Operations Team  
**DATE:** October 3, 2026  
**SUBJECT:** Ticket classification and routing analysis (Jan 2025 – Jun 2026)  

---

Hi Priya,

Here is a summary of what we found after analyzing our support ticket data, where our queue bottlenecks actually are, and what this means for our next two support hires.

### 1. Why we did this
We know our intake chatbot tags are often wrong, which leads to tickets getting sent to the wrong queues, bounced between agents, and delayed. We built a classification model to review customer messages and agent closing notes against Support Operating Policy v3.2. The goal was to see where tickets should have gone, estimate the friction this causes, and give leadership clear evidence for capacity planning.

### 2. Headline numbers
We analyzed **11,641 tickets** between **January 1, 2025 and June 30, 2026**:
- **3,020 tickets (25.9%)** were assigned by the intake bot to a different team than the model predicted.
- Our support policy (§4) sets an internal benchmark of **₹305 per transfer**.
- Multiplying 3,020 misrouted tickets by ₹305 gives **₹921,100 (~₹9.21 lakh)** in estimated potential process friction.
- *Important note:* This figure represents estimated operational waste and re-handling friction across teams. It is **not cash savings or money we can put in the bank**, because not every misrouted ticket was formally re-transferred in the system.

### 3. What the queues actually look like
When we compare where the chatbot sent tickets against where the model shows the work actually belongs, there is a clear mismatch:

| Support Team | Bot Assigned | Model Predicted | Difference | Share of Workload | Operational Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistics** | 1,905 | **3,334** | **+1,429 (+75.0%)** | **28.6%** | Tier 1 / Specialized |
| **Chat Frontline** | 3,030 | **2,308** | -722 (-23.8%) | 19.8% | Tier 1 |
| **Returns Desk** | 1,049 | **1,758** | **+709 (+67.6%)** | 15.1% | Tier 1 / Specialized |
| **Billing** | 2,425 | **1,601** | **-824 (-34.0%)** | 13.8% | Tier 1 / Specialized |
| **Email Frontline** | 1,807 | **1,374** | -433 (-24.0%) | 11.8% | Tier 1 |
| **Voice Frontline** | 900 | **710** | -190 (-21.1%) | 6.1% | Tier 1 |
| **Escalations & Warranty** | 525 | **556** | +31 (+5.9%) | 4.8% | **Tier 2 (Certified RMA)** |

Two major patterns stand out:
1. **The Billing queue was artificially inflated:** The chatbot made Billing look like our second-largest queue (2,425 tickets). But **46.3% of those tickets were actually shipping delays or return queries** where customers happened to write words like "paid" or "money debited". True Billing workload is closer to 13.8% (1,601 tickets).
2. **Logistics is carrying the heaviest load:** Logistics handles about **28.6% of total demand** (3,334 tickets). This explains why that team has been overwhelmed and facing longer resolution times.

### 4. Human validation check
To verify the model without circular assumptions, a human reviewer evaluated a blind sample of **30 tickets** (`seed=42`) with AI predictions hidden:
- **Category agreement:** **30 / 30 tickets (100.0%)** between human reviewer and model.
- **Team agreement:** **29 / 30 tickets (96.7%)**.
- **Original bot tag vs. human reviewer:** **21 / 30 tickets (70.0%)**.
- **The single discrepancy:** Ticket `TK-247547` mentioned audio crackling alongside a warranty question. The model tagged it for frontline troubleshooting, while the human reviewer routed it directly to Tier 2 Warranty.
- *Caveat:* These figures show agreement on this specific 30-ticket sample. The sample is **directional and not sufficient to estimate production-wide accuracy** across all 11,641 tickets.

### 5. Staffing guidance: where the two hires should go
While Logistics has the highest volume, we should **not automatically put both new hires into Logistics**:
1. **Volume vs. Complexity:** Tier 1 and Tier 2 workloads cannot be compared by ticket count alone. Under Support Policy §6, **Escalations & Warranty (Tier 2)** handles certified RMA investigations and vendor repairs. Those cases take days, not minutes. Even though Tier 2 is only 4.8% of volume (556 tickets), each case requires certified technical expertise.
2. **Fix intake routing first:** If we update the chatbot's keyword triggers so it stops dumping delivery issues into Billing, we can remove a significant amount of hand-off friction without adding headcount.

### 6. Limitations and next steps
- **Keep in mind:** The model provides statistical estimates based on text patterns, not absolute truth. When a customer reports multiple issues at once (e.g., hardware fault plus refund request), single-label routing can miss context. Also, older Freshdesk tickets do not contain transfer counts.
- **Recommended next steps:**
  1. Fix the chatbot's intake logic so shipping complaints with payment words stop going to Billing.
  2. Decide on the two hires by balancing Logistics volume against Tier 2 technical complexity.
  3. Keep reviewing a small sample of tickets each month to monitor how routing improves.

Let me know if you would like to review the dashboard together.
