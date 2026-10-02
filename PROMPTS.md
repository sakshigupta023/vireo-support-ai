# AI Prompts & Iteration History — Vireo Audio Support AI

This document details the exact sequence of prompts provided during the development of the Vireo Audio Support Intelligence system, what each iteration produced, the analytical limitations identified, and the subsequent audit refinements.

---

## 1. Initial Prompt (V1 Request)

### Prompt Text
> We need to complete a take-home assignment for Vireo Audio.
> 
> IMPORTANT: Do not start by building a huge application. First inspect ALL provided files and understand the business problem.
> 
> Read all of these files in the project folder:
> - README.txt
> - email-thread.txt
> - support-policy.pdf
> - tickets.csv
> - agents.csv
> - customers.csv
> - orders.csv
> - products.csv
> 
> Task context:
> Vireo Audio wants monthly support-ticket volume by category and team and wants to understand where the next two support hires should go. Their existing category tags are known to be unreliable.
> 
> Build a small Python + Streamlit proof of concept.
> 
> For V1, implement only:
> 1. Load tickets.csv and the reference CSV files.
> 2. Show basic dataset statistics.
> 3. Preserve the existing ticket category.
> 4. Create an AI-assisted classification pipeline that predicts the appropriate existing support category from the customer message and/or agent closing note.
> 5. Output: ticket_id, original category, predicted category, predicted team, confidence, short reason.
> 6. Show monthly ticket volume by category and assigned team.
> 7. Identify potential category/team misrouting.
> 8. Calculate transfer-related cost using ₹305 internal transfer cost from support-policy.pdf.
> 9. Do NOT treat the existing category as ground truth.
> 10. Do NOT invent business metrics or conclusions.
> 
> Use a simple, maintainable architecture.
> 
> Create:
> - app.py
> - classifier.py
> - analysis.py
> - evaluate.py
> - requirements.txt
> - README.md
> - .gitignore

### What V1 Produced
- **File Structure:** Created `app.py`, `classifier.py`, `analysis.py`, `evaluate.py`, `requirements.txt`, `README.md`, and `.gitignore`.
- **Hybrid AI Classifier:** Implemented `TicketClassifier` in `classifier.py` combining regex domain rules with a TF-IDF + Logistic Regression statistical model to predict the 11 categories and map them to the 7 operational teams.
- **Analytics Engine:** Built `analysis.py` to aggregate monthly volume by category and team, detect mismatches against initial bot tags, and calculate transfer costs at ₹305/transfer.
- **CLI & Streamlit Dashboard:** Built a 5-tab Streamlit dashboard and a terminal-based evaluation runner in `evaluate.py`.

### Problems Identified in V1
1. **Conflating Predictions with Ground Truth:** V1 outputs were occasionally referenced as "true workload" or "ground truth" rather than probabilistic model predictions.
2. **Definitive Staffing Mandate:** V1 stated that hires *"must go to Logistics"* rather than presenting observed workload differences for leadership decision-making.
3. **Lack of a Validated Audit Sample:** Misrouting claims lacked an independent, manually audited baseline sample to verify classifier accuracy.
4. **Oversimplified Transfer Cost Claim:** Framing ₹305 transfer costs as "guaranteed savings" rather than an estimated operational process-cost exposure.
5. **Overlooking Tier 2 Complexity:** V1 compared Tier 2 (`Escalations & Warranty`) solely on raw ticket volume without citing Support Policy §6 (Tier 2 cases are certified, multi-touch RMA investigations measured in resolution days).

---

## 2. Refinement & Audit Prompt (V2 Request)

### Prompt Text
> V1 is built, but before we treat it as final, audit the analysis and make the conclusions evidence-based.
> 
> Important corrections:
> 1. Do NOT call the model output "ground truth". The AI predictions are predictions, not verified ground truth.
> 2. Do NOT state "the two hires should go to Logistics" as a definitive decision. Instead, present the observed workload differences and let the business owner make the staffing decision.
> 3. Do NOT claim that a specific number of tickets were "misrouted" unless there is a clearly defined validation method supporting that claim.
> 4. Create a manual validation sample:
>    - Randomly select 100 tickets, using a fixed random seed.
>    - Show customer_message + agent_notes + original category + predicted category + predicted team.
>    - Allow/prepare a way to manually review whether the prediction is reasonable.
>    - Calculate agreement/error rate from that reviewed sample.
>    - Report the main types of wrong predictions.
> 5. Clearly separate: Original bot category, AI prediction, Manually reviewed/validated label, Observed team workload.
> 6. Keep the ₹305 internal transfer cost calculation, but clearly label it as an estimated process-cost opportunity based on observed transfers/misrouting assumptions, not guaranteed savings.
> 7. Follow the support policy:
>    - Tier 2 teams should not be directly compared with Tier 1 teams purely by ticket volume.
>    - Use assigned/resolving team definitions carefully.
>    - Do not treat blank legacy transfers as zero.
> 8. Remove unsupported language (e.g. "Empirical Ground Truth", "must go to Logistics", "will recover ₹X") and replace with factual wording.
> 9. Add a clear "Limitations & Known Issues" section to the dashboard.
> 10. Add a PROMPTS.md file documenting the AI prompts used in this project.
> 11. Update README.md so that a new person can understand tool usage, classification, validation, limitations, and offline execution.
> 12. Run evaluate.py again after these changes and show the actual validation results.

### What Changed in V2
1. **Manual Validation Audit:** Generated a fixed 100-ticket audit sample (`validation_sample_100.csv`, `random_state=42`) with independent expert review, measuring **95.0% AI-to-Auditor agreement** and **65.0% Bot-to-Auditor agreement**, with an explicit breakdown of the 5 error edge cases.
2. **Clear Label Separation:** Strictly delineated Original Bot Category, AI Prediction, Manually Audited Label, and Observed Team Workload across code and UI.
3. **Objective Decision Framing:** Restructured conclusions to present observed workload differences (Logistics at 29.5% model workload vs Billing at 13.6%), contextualized with Tier 2 certified RMA requirements (§6).
4. **Precise Cost Framing:** Reframed transfer calculations as *estimated process-cost exposure*, noting legacy transfer missingness (§9).
5. **Dashboard & Documentation Expansion:** Added dedicated "Manual Validation Audit" and "Limitations & Known Issues" tabs in `app.py`, created `PROMPTS.md`, and updated `README.md` and `evaluate.py`.

---

## 3. Methodology Audit & Genuine Validation Prompt (V3 Request)

### Discovery of Circular Validation in V2
During an internal audit of the validation methodology, the script that generated `validation_sample_100.csv` (`scratch/generate_validated_sample.py`) was inspected. The audit revealed:
```python
val_cat = pred  # Initialized directly from model prediction
if "rma" in msg or "rma" in notes:
    val_cat = "Warranty & Repair"
# ... minor keyword heuristics ...
```
Because the "auditor" labels were seeded with the model's own predictions and modified only by heuristic overrides, the 95.0% agreement metric was **inherently circular, artificial, and invalid**. It was not an independent check.

### Prompt Text (V3 Request)
> The previous 95% validation metric is invalid because the benchmark labels were generated from the model's own predictions.
> We need to fix this properly. Do NOT claim 95% accuracy or independent validation.
> 
> Proceed with V3, but use a 30-ticket manual validation sample instead of 100 because we need a fast, genuinely human-reviewed validation.
> 
> Implement:
> 1. Main analysis period: Jan 1, 2025 through Jun 30, 2026 inclusive.
> 2. Select exactly 30 tickets from that period using random seed=42.
> 3. Create a manual validation workflow in the Streamlit app.
>    Before review, show only: ticket_id, customer_message, agent_notes, original_category.
>    Do NOT show AI prediction before the human submits their review.
> 4. Human reviewer selects: human_validated_category, human_validated_team, reviewer_notes.
>    Do NOT prefill these fields with AI predictions. Do NOT generate them with heuristics.
> 5. After submission, reveal: AI predicted category, AI predicted team, category match/mismatch, team match/mismatch.
> 6. Save reviews to: manual_validation_reviews.csv.
> 7. Show: Reviewed: X/30, Remaining: X.
>    If no reviews: "Manual validation pending — no accuracy claim available."
>    Once reviews exist: actual agreement/error rate based ONLY on completed human reviews.
> 8. Remove the old circular 95% metric and all wording claiming it was independently reviewed.
> 9. Keep the old circular validation file only as historical/audit evidence if needed, but ensure no final analysis uses it.
> 10. Update README.md and PROMPTS.md to document: the old 95% metric was discarded, why it was circular, the new human-review workflow, limitations of a 30-ticket sample.
> 11. Run: .\.venv\Scripts\python.exe evaluate.py.
> 12. Run the Streamlit app and verify the manual review page works.

### What Changed in V3
1. **Discarded Circular Metric:** The 95% heuristic metric was permanently removed from `app.py`, `evaluate.py`, `analysis.py`, and `README.md`. `validation_sample_100.csv` was preserved strictly as historical audit evidence.
2. **Analysis Period Alignment:** Standardized primary analysis on **Jan 1, 2025 through Jun 30, 2026 inclusive** (11,641 tickets).
3. **Reproducible 30-Ticket Sample:** Extracted exactly 30 tickets (`random_state=42`) from the analysis period and generated `manual_validation_reviews.csv` with empty human validation fields (no AI prefill, no regex heuristics).
4. **Blinded Streamlit Review UI (Tab 3):**
   - Reviewers inspect `ticket_id`, `channel`, `customer_message`, `agent_notes`, and `original_category`.
   - AI predictions are **strictly blinded** prior to review submission.
   - Category and Team selectors have no prefilled defaults.
   - Upon submission, reviews are saved to `manual_validation_reviews.csv`, and the UI reveals AI predictions, category match/mismatch, and team match/mismatch.
5. **Conditional Accuracy Reporting:**
   - With 0 reviews: Displays `"Manual validation pending — no accuracy claim available."`
   - With $X > 0$ reviews: Computes genuine agreement and error rates strictly over the completed human reviews.
6. **CLI Audit Alignment:** Updated `evaluate.py` to inspect `manual_validation_reviews.csv` and report validation progress or pending review status.
7. **Documented Limitations:** Cites the statistical trade-offs of a 30-ticket sample size ($\pm 10\text{–}15\%$ margin of error) as a directional, qualitative assessment tool.

---

## 4. Final Review & Terminology Precision (V3 Polish)

### Prompt Text
> Review the current Vireo Audio Task 1 project against the assignment requirements.
> Do NOT change the core analysis logic or invent new results.
> Update the README to reflect the final V3 state accurately:
> 1. Remove/avoid calling the project "V1 Proof of Concept" in the main title.
> 2. Clearly state the analysis period: Jan 1 2025 to Jun 30 2026.
> 3. Document the final human-validation results:
>    - 30/30 category agreement (100%)
>    - 29/30 responsible-team agreement (96.7%)
>    - Original intake category vs human: 21/30 (70%)
> 4. Explicitly describe these as agreement on a 30-ticket human-reviewed sample, NOT production accuracy.
> 5. Document the 3,020 model-vs-original team routing discrepancies and ₹921,100 (₹9.21 lakh) estimated process-cost exposure at ₹305 per transfer-equivalent.
> 6. Clearly state that this is potential process friction, not guaranteed savings.
> 7. Document the limitations already present in the project.
> 8. Keep the existing methodology and historical discarded-validation explanation.
> 9. Do not add unsupported claims.
> 
> Make two small corrections only:
> 1. In README.md, remove the specific statement claiming an "expected statistical margin of error (±10–15%)" for the N=30 validation sample. Replace it with a simple statement that the 30-ticket sample is directional and not sufficient to estimate production-wide accuracy.
> 2. Replace wording such as "Auditor vs. AI" with "Human reviewer vs. AI" or "Human validation vs. AI" wherever it refers to the 30-ticket manual review, because the labels were independently reviewed by a human reviewer.

### What Changed in this Iteration
1. **Terminology Standardization:** Updated documentation and CLI/app display strings from "Auditor" to **"Human Reviewer"** across `README.md`, `evaluate.py`, and `app.py` to accurately reflect the independent human-review process.
2. **Removed Unsupported Margin-of-Error Claim:** Eliminated the unsupported "$\pm 10\text{–}15\%$ statistical margin-of-error" statement from `README.md` and `app.py`.
3. **Explicit Directional Scope Limitation:** Clarified across documentation and UI that the 30-ticket sample is **directional and not sufficient to estimate production-wide accuracy**, serving instead as a qualitative validation of model classification and edge-case handling.
4. **Consistency Updates:** Updated `evaluate.py` console output and `app.py` UI labels to maintain strict consistency with these reporting guidelines.
5. **Execution Verification:** Re-ran `evaluate.py` to confirm terminal output and verified the final audit results on the completed 30-ticket human review:
   - **Human Reviewer vs. AI Category Agreement:** **30 / 30 (100.0%)**
   - **Human Reviewer vs. AI Team Agreement:** **29 / 30 (96.7%)**
   - **Original Bot Tag vs. Human Reviewer:** **21 / 30 (70.0%)**


