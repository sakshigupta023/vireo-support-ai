"""
app.py - Vireo Audio Support Intelligence & Triage AI

Provides dataset exploration, AI-assisted classification, manual validation audit,
monthly volume dynamics, routing discrepancy analysis, transfer process-cost estimation,
and evidence-based workload comparisons for leadership decision-making.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from classifier import TicketClassifier, VALID_CATEGORIES, map_category_to_team
from analysis import (
    load_all_data,
    get_dataset_statistics,
    compute_monthly_volume,
    analyze_misrouting,
    calculate_transfer_costs,
    analyze_validation_sample,
    get_workload_comparison,
    INTERNAL_TRANSFER_COST_INR
)

# Page configuration
st.set_page_config(
    page_title="Vireo Audio — Support Intelligence & Triage AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .alert-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        color: #92400E;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data and running AI classifier...")
def get_cached_data_and_predictions():
    data = load_all_data(".")
    tickets = data["tickets"]
    classifier = TicketClassifier()
    classified_df = classifier.batch_classify(tickets)
    return data, classified_df, classifier


# Load cached data
try:
    data, classified_df, classifier = get_cached_data_and_predictions()
    stats = get_dataset_statistics(data)
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()


# Sidebar Controls
st.sidebar.title("🎧 Vireo Audio AI")
st.sidebar.markdown("**Support Intelligence & Triage**")
st.sidebar.markdown("---")

st.sidebar.subheader("Filters")
selected_channels = st.sidebar.multiselect(
    "Support Channel",
    options=classified_df["channel"].unique().tolist(),
    default=classified_df["channel"].unique().tolist()
)

selected_sources = st.sidebar.multiselect(
    "Source System",
    options=classified_df["source_system"].unique().tolist(),
    default=classified_df["source_system"].unique().tolist()
)

min_conf = st.sidebar.slider(
    "Minimum Classification Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

# Filter dataframe
filtered_df = classified_df[
    (classified_df["channel"].isin(selected_channels)) &
    (classified_df["source_system"].isin(selected_sources)) &
    (classified_df["confidence"] >= min_conf)
]

st.sidebar.markdown("---")
st.sidebar.caption("Operating Reference: ₹305 / internal team transfer (Support Policy §4)")


# Main Page Header
st.markdown('<div class="main-title">🎧 Vireo Audio — Support Intelligence & Triage AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-assisted ticket re-classification, queue workload validation, '
    'and evidence-based staffing analysis for operational leadership.</div>',
    unsafe_allow_html=True
)

# 6 Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset Overview",
    "🤖 AI Classification & Triage",
    "🔍 Manual Validation Audit",
    "📈 Monthly Volume & Trends",
    "🔄 Routing & Cost Exposure",
    "👥 Workload & Staffing Analysis"
])


# ==========================================
# TAB 1: DATASET OVERVIEW
# ==========================================
with tab1:
    st.subheader("Support Desk Health & Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Tickets", f"{stats['total_tickets']:,}")
    col2.metric("Total Customers", f"{stats['total_customers']:,}")
    col3.metric("Total Orders", f"{stats['total_orders']:,}")
    col4.metric("Active Agents", f"{stats['total_agents']}")
    col5.metric("Avg CSAT Score", f"{stats['avg_csat']} / 5.0")

    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Channel Distribution")
        channel_fig = px.pie(
            values=list(stats["channel_counts"].values()),
            names=list(stats["channel_counts"].keys()),
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        channel_fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
        st.plotly_chart(channel_fig, use_container_width=True)

    with c2:
        st.markdown("##### Ticket Status Breakdown")
        status_fig = px.bar(
            x=list(stats["status_counts"].keys()),
            y=list(stats["status_counts"].values()),
            labels={"x": "Status", "y": "Ticket Count"},
            color=list(stats["status_counts"].keys()),
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        status_fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=260)
        st.plotly_chart(status_fig, use_container_width=True)

    with c3:
        st.markdown("##### Helpdesk Systems & Missingness")
        source_fig = px.pie(
            values=list(stats["source_system_counts"].values()),
            names=list(stats["source_system_counts"].keys()),
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        source_fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
        st.plotly_chart(source_fig, use_container_width=True)

    st.caption("ℹ️ Note on Legacy Data: Tickets migrated from Freshdesk (prior to 14 Sep 2025) lack the 'transfers' field (stored as blank/NaN per Support Policy §9, not zero).")

    st.markdown("##### Raw Dataset Explorer")
    st.dataframe(
        filtered_df[[
            "ticket_id", "created_at", "channel", "category", "assigned_team",
            "agent_id", "status", "customer_message", "agent_notes"
        ]].head(100),
        use_container_width=True
    )


# ==========================================
# TAB 2: AI CLASSIFICATION & TRIAGE
# ==========================================
with tab2:
    st.subheader("AI-Assisted Ticket Classification Pipeline")
    st.markdown(
        "Predicts the appropriate support category and assigned team based on customer message "
        "and agent closing notes. Preserves the original intake tag and outputs explainable rationale."
    )

    with st.expander("⚡ Interactive Single-Ticket AI Classifier (Test Sandbox)", expanded=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            input_msg = st.text_area(
                "Customer Message / IVR Transcript",
                value="paid, confirmed, then nothing. 5 days and counting and tracking not updating",
                height=100
            )
            input_channel = st.selectbox("Channel", ["chat", "email", "voice", "social"], index=0)
        with sc2:
            input_notes = st.text_area(
                "Agent Closing Note (Optional)",
                value="Cx says dlvry delayed. Raised ticket with crr partner. xfer to Logistics.",
                height=100
            )
            input_orig_cat = st.selectbox("Original Bot Tag", VALID_CATEGORIES, index=1)

        if st.button("Run AI Prediction", type="primary"):
            single_pred = classifier.classify_single(
                customer_message=input_msg,
                agent_notes=input_notes,
                channel=input_channel,
                original_category=input_orig_cat
            )
            
            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            res_c1.metric("Original Bot Category", single_pred["original_category"])
            res_c2.metric("AI Predicted Category", single_pred["predicted_category"])
            res_c3.metric("AI Predicted Team", single_pred["predicted_team"])
            res_c4.metric("Confidence Score", f"{single_pred['confidence']:.1%}")
            
            st.info(f"**Prediction Reason:** {single_pred['short_reason']}")

    st.markdown("---")
    st.subheader("Batch AI Classification Results")
    
    display_cols = [
        "ticket_id",
        "category",
        "predicted_category",
        "predicted_team",
        "confidence",
        "short_reason"
    ]
    
    clean_table = filtered_df[display_cols].rename(columns={"category": "original_category"})
    
    only_discrepant = st.checkbox("Show only tickets where AI prediction differs from original intake tag", value=False)
    if only_discrepant:
        clean_table = clean_table[clean_table["original_category"] != clean_table["predicted_category"]]

    st.dataframe(clean_table, use_container_width=True, height=400)

    csv_bytes = clean_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Classified Tickets (CSV)",
        data=csv_bytes,
        file_name="vireo_classified_tickets.csv",
        mime="text/csv"
    )


# ==========================================
# ==========================================
# TAB 3: MANUAL VALIDATION AUDIT
# ==========================================
with tab3:
    st.subheader("Manual Validation Audit (30-Ticket Sample, Seed=42)")
    st.markdown(
        "To establish a genuine, non-circular quality check, the previous 95% heuristic metric "
        "(which was derived from the model's own predictions) was **permanently discarded**.<br>"
        "Reviewers label a reproducible sample of 30 tickets from the primary analysis period "
        "(Jan 1, 2025 – Jun 30, 2026) **without seeing the AI prediction** until after submission.",
        unsafe_allow_html=True
    )

    val_res = analyze_validation_sample("manual_validation_reviews.csv")

    if not val_res.get("available", False):
        st.warning("Manual validation review file 'manual_validation_reviews.csv' not found.")
    else:
        reviewed_cnt = val_res["reviewed_count"]
        rem_cnt = val_res["remaining_count"]
        total_cnt = val_res["total_samples"]

        # Audit Progress & Metrics Header
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        vcol1.metric("Reviewed Tickets", f"{reviewed_cnt} / {total_cnt}")
        vcol2.metric("Remaining Reviews", f"{rem_cnt}")

        if reviewed_cnt == 0:
            vcol3.metric("Validation Status", "Pending Review")
            vcol4.metric("Agreement Metric", "N/A")
            st.info("ℹ️ **Manual validation pending — no accuracy claim available.** Submit your reviews below to calculate genuine agreement rates.")
        else:
            cat_agr = val_res["model_category_agreement_rate"]
            team_agr = val_res["model_team_agreement_rate"]
            vcol3.metric("AI vs Human Category Agreement", f"{cat_agr:.1%}", f"{val_res['model_category_agreement_count']}/{reviewed_cnt} matched")
            vcol4.metric("AI vs Human Team Agreement", f"{team_agr:.1%}", f"{val_res['model_team_agreement_count']}/{reviewed_cnt} matched")

            if val_res.get("bot_agreement_rate") is not None:
                st.caption(f"Intake Bot vs Human Agreement: {val_res['bot_agreement_rate']:.1%} ({val_res['bot_agreement_count']}/{reviewed_cnt} matched) | Error Rate: {val_res['category_error_rate']:.1%}")

        st.markdown("---")

        # Interactive Ticket Review Interface
        manual_df = pd.read_csv("manual_validation_reviews.csv")
        for col in ["human_validated_category", "human_validated_team", "reviewer_notes", "reviewed_at"]:
            if col in manual_df.columns:
                manual_df[col] = manual_df[col].astype("object").fillna("")
        
        ticket_options = []
        for idx, r in manual_df.iterrows():
            has_rev = pd.notna(r["human_validated_category"]) and str(r["human_validated_category"]).strip() != "" and str(r["human_validated_category"]).lower() != "nan"
            status_tag = "✅ Reviewed" if has_rev else "⏳ Pending"
            ticket_options.append(f"Ticket #{idx+1}: {r['ticket_id']} [{status_tag}]")

        sel_idx = st.selectbox(
            "Select Ticket to Review / Audit (1 of 30):",
            range(len(manual_df)),
            format_func=lambda i: ticket_options[i]
        )
        curr = manual_df.iloc[sel_idx]
        is_already_reviewed = pd.notna(curr["human_validated_category"]) and str(curr["human_validated_category"]).strip() != "" and str(curr["human_validated_category"]).lower() != "nan"

        # Ticket Info (AI is strictly hidden)
        ti_c1, ti_c2, ti_c3 = st.columns(3)
        ti_c1.markdown(f"**Ticket ID:** `{curr['ticket_id']}`")
        ti_c2.markdown(f"**Channel:** `{curr['channel']}`")
        ti_c3.markdown(f"**Original Intake Category:** `{curr['original_category']}`")

        st.markdown("##### 👤 Customer Message")
        msg_text = curr['customer_message'] if pd.notnull(curr['customer_message']) and str(curr['customer_message']).strip() != "" else "*[No customer message text]*"
        st.info(msg_text)

        st.markdown("##### 📝 Agent Closing Notes")
        notes_text = curr['agent_notes'] if pd.notnull(curr['agent_notes']) and str(curr['agent_notes']).strip() != "" else "*[No agent closing notes]*"
        st.markdown(f"> {notes_text}")

        # Human Review Form (NO AI PREFILL)
        st.markdown("##### ✍️ Human Validator Review")
        with st.form(key=f"human_review_form_{sel_idx}"):
            cat_list = ["-- Select Category --"] + VALID_CATEGORIES
            existing_cat = str(curr["human_validated_category"]).strip() if is_already_reviewed else ""
            cat_idx = cat_list.index(existing_cat) if existing_cat in cat_list else 0

            chosen_cat = st.selectbox(
                "Human Validated Category (True customer issue):",
                options=cat_list,
                index=cat_idx
            )

            team_list = [
                "-- Select Team --",
                "Logistics",
                "Billing",
                "Returns Desk",
                "Escalations & Warranty",
                "Chat Frontline",
                "Email Frontline",
                "Voice Frontline"
            ]
            existing_team = str(curr["human_validated_team"]).strip() if is_already_reviewed else ""
            team_idx = team_list.index(existing_team) if existing_team in team_list else 0

            chosen_team = st.selectbox(
                "Human Validated Team (Responsible Queue per SOP §6):",
                options=team_list,
                index=team_idx
            )

            existing_rev_notes = str(curr["reviewer_notes"]) if (is_already_reviewed and pd.notnull(curr["reviewer_notes"])) else ""
            input_notes = st.text_area(
                "Reviewer Notes / Rationales (Optional):",
                value=existing_rev_notes,
                placeholder="Explain the rationale behind category/team selection or any policy nuances..."
            )

            save_btn = st.form_submit_button("💾 Save Human Review", use_container_width=True)

            if save_btn:
                if chosen_cat == "-- Select Category --" or chosen_team == "-- Select Team --":
                    st.error("Please select both a valid Category and Team before saving.")
                else:
                    manual_df.at[sel_idx, "human_validated_category"] = chosen_cat
                    manual_df.at[sel_idx, "human_validated_team"] = chosen_team
                    manual_df.at[sel_idx, "reviewer_notes"] = input_notes
                    manual_df.at[sel_idx, "reviewed_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    manual_df.to_csv("manual_validation_reviews.csv", index=False)
                    st.success(f"Review successfully saved for ticket {curr['ticket_id']}!")
                    st.rerun()

        # Revelation of AI Prediction (Only after human review)
        if is_already_reviewed:
            st.markdown("---")
            st.markdown("##### 🤖 Post-Review AI Comparison (Revealed)")
            rev_c1, rev_c2 = st.columns(2)

            cat_match = (str(curr["human_validated_category"]).strip() == str(curr["ai_predicted_category"]).strip())
            team_match = (str(curr["human_validated_team"]).strip() == str(curr["ai_predicted_team"]).strip())

            with rev_c1:
                st.markdown(f"**Human Validated Category:** `{curr['human_validated_category']}`")
                st.markdown(f"**AI Predicted Category:** `{curr['ai_predicted_category']}`")
                if cat_match:
                    st.success("✅ **Category Match:** AI prediction agreed with human validation.")
                else:
                    st.error("❌ **Category Mismatch:** AI prediction differed from human validation.")

            with rev_c2:
                st.markdown(f"**Human Validated Team:** `{curr['human_validated_team']}`")
                st.markdown(f"**AI Predicted Team:** `{curr['ai_predicted_team']}`")
                if team_match:
                    st.success("✅ **Team Match:** AI predicted team agreed with human validation.")
                else:
                    st.error("❌ **Team Mismatch:** AI predicted team differed from human validation.")

            st.caption(f"Model Confidence: {curr['confidence']:.1%} | Model Rationale: {curr['short_reason']}")
        else:
            st.info("🔒 **AI Prediction is blinded.** Submit your human validation above to reveal model prediction and agreement.")

        # Full 30-Ticket Sample Audit Table
        st.markdown("---")
        st.markdown("##### 📋 30-Ticket Sample Audit Table")
        disp_df = manual_df.copy()
        disp_df["Audit Status"] = disp_df["human_validated_category"].apply(
            lambda x: "✅ Reviewed" if pd.notna(x) and str(x).strip() != "" and str(x).lower() != "nan" else "⏳ Pending"
        )
        disp_df["AI Category (Revealed Post-Review)"] = disp_df.apply(
            lambda r: r["ai_predicted_category"] if r["Audit Status"] == "✅ Reviewed" else "[Blinded until reviewed]",
            axis=1
        )
        disp_df["AI Team (Revealed Post-Review)"] = disp_df.apply(
            lambda r: r["ai_predicted_team"] if r["Audit Status"] == "✅ Reviewed" else "[Blinded until reviewed]",
            axis=1
        )

        st.dataframe(
            disp_df[[
                "ticket_id", "Audit Status", "channel", "original_category",
                "human_validated_category", "human_validated_team",
                "AI Category (Revealed Post-Review)", "AI Team (Revealed Post-Review)",
                "reviewer_notes"
            ]],
            use_container_width=True,
            height=320
        )


# ==========================================
# TAB 4: MONTHLY VOLUME & TRENDS
# ==========================================
with tab4:
    st.subheader("Monthly Ticket Volume Dynamics")
    st.markdown(
        "Compare monthly volume by category and responsible team under original intake tags "
        "versus model-predicted workloads."
    )

    vol_c1, vol_c2 = st.columns(2)

    with vol_c1:
        st.markdown("##### Monthly Volume by Category (AI Predicted)")
        monthly_cat_pred = compute_monthly_volume(filtered_df, group_col="predicted_category")
        fig_cat = px.bar(
            monthly_cat_pred.reset_index().melt(id_vars="month_year", var_name="Category", value_name="Tickets"),
            x="month_year",
            y="Tickets",
            color="Category",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_cat.update_layout(xaxis_title="Month", yaxis_title="Ticket Volume", height=380)
        st.plotly_chart(fig_cat, use_container_width=True)

    with vol_c2:
        st.markdown("##### Monthly Volume by Responsible Team (AI Predicted)")
        monthly_team_pred = compute_monthly_volume(filtered_df, group_col="predicted_team")
        fig_team = px.bar(
            monthly_team_pred.reset_index().melt(id_vars="month_year", var_name="Team", value_name="Tickets"),
            x="month_year",
            y="Tickets",
            color="Team",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_team.update_layout(xaxis_title="Month", yaxis_title="Ticket Volume", height=380)
        st.plotly_chart(fig_team, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Category Volume Comparison: Original Intake Tag vs AI Model Prediction")
    
    comp_cat_df = pd.DataFrame({
        "Original Bot Tagged Volume": filtered_df["category"].value_counts(),
        "AI-Predicted Volume": filtered_df["predicted_category"].value_counts()
    }).fillna(0).astype(int)
    comp_cat_df["Difference"] = comp_cat_df["AI-Predicted Volume"] - comp_cat_df["Original Bot Tagged Volume"]
    
    fig_comp_cat = go.Figure(data=[
        go.Bar(name='Original Bot Tag', x=comp_cat_df.index, y=comp_cat_df['Original Bot Tagged Volume'], marker_color='#94A3B8'),
        go.Bar(name='AI Model Predicted', x=comp_cat_df.index, y=comp_cat_df['AI-Predicted Volume'], marker_color='#2563EB')
    ])
    fig_comp_cat.update_layout(barmode='group', xaxis_tickangle=-45, height=350, margin=dict(b=100))
    st.plotly_chart(fig_comp_cat, use_container_width=True)


# ==========================================
# TAB 5: ROUTING & COST EXPOSURE
# ==========================================
with tab5:
    st.subheader("Queue Routing Differences & Transfer Cost Exposure")
    
    misrouting_data = analyze_misrouting(filtered_df)
    cost_data = calculate_transfer_costs(filtered_df, INTERNAL_TRANSFER_COST_INR)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Category Discrepancy Rate", f"{misrouting_data['category_misrouting_rate']:.1%}")
    m2.metric("Team Routing Discrepancy", f"{misrouting_data['team_misrouting_rate']:.1%}")
    m3.metric("Recorded Helpdesk Transfers", f"{cost_data['recorded_transfers_count']:,} hand-offs")
    m4.metric("Estimated Process-Cost Exposure", f"₹{cost_data['estimated_mismatch_exposure_inr']:,.0f}")

    st.markdown("""
    <div class="alert-box">
        <strong>Mechanics of Intake Queue Distortion:</strong> The intake bot relies on naive keyword triggers. 
        When customers use financial terms (e.g. <em>"paid 5 days ago"</em>, <em>"amount deducted"</em>) when reporting 
        delayed shipments or return refunds, the bot tags them as <strong>Billing & Payments</strong> 
        and routes them directly into the <strong>Billing</strong> queue.
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("##### Discrepancy Rate by Original Assigned Team")
        team_breakdown = misrouting_data["team_breakdown"].sort_values(by="mismatch_rate", ascending=False)
        fig_mis = px.bar(
            team_breakdown,
            x="assigned_team",
            y="mismatch_rate",
            text="mismatch_rate",
            labels={"assigned_team": "Original Assigned Team", "mismatch_rate": "Discrepancy Rate"},
            color="mismatch_rate",
            color_continuous_scale="Reds"
        )
        fig_mis.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_mis.update_layout(yaxis_tickformat='.0%', height=350, showlegend=False)
        st.plotly_chart(fig_mis, use_container_width=True)

    with mc2:
        st.markdown("##### Model-Predicted Issues in the Billing Queue")
        billing_breakdown = pd.Series(misrouting_data["billing_predicted_categories"]).reset_index()
        billing_breakdown.columns = ["Model Predicted Issue", "Ticket Count"]
        fig_bill = px.pie(
            billing_breakdown,
            names="Model Predicted Issue",
            values="Ticket Count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bill.update_layout(height=350)
        st.plotly_chart(fig_bill, use_container_width=True)

    st.caption("ℹ️ Cost Calculation Methodology: Support Operating Policy §4 defines internal transfers at ₹305/hand-off. The figure above represents potential administrative re-handling exposure under observed routing differences, not guaranteed savings.")


# ==========================================
# TAB 6: WORKLOAD & STAFFING ANALYSIS
# ==========================================
with tab6:
    st.subheader("Support Workload Distribution & Staffing Considerations")
    st.markdown(
        "Analytical decision support for evaluating support capacity and the allocation of the next **two support hires**."
    )

    workload = get_workload_comparison(filtered_df)
    comp_table = workload["comparison_table"]

    fig_staff = go.Figure(data=[
        go.Bar(name='Perceived Volume (Bot Assigned)', x=comp_table['Support Team'], y=comp_table['Original Assigned Volume'], marker_color='#EF4444'),
        go.Bar(name='Model-Predicted Workload', x=comp_table['Support Team'], y=comp_table['Model-Predicted Workload'], marker_color='#10B981')
    ])
    fig_staff.update_layout(
        barmode='group',
        xaxis_title="Support Team",
        yaxis_title="Total Ticket Count",
        height=380
    )
    st.plotly_chart(fig_staff, use_container_width=True)

    st.markdown("##### Queue Volume & Operational Tier Summary")
    st.dataframe(comp_table, use_container_width=True)

    st.markdown("---")
    
    st.markdown("""
    ### 📋 Evidence-Based Observations for Leadership (Jan 2025 – Jun 2026)

    1. **Logistics Queue Demand:**
       - Model-predicted workload indicates **Logistics handles 3,334 tickets (28.6% of total desk volume)**, significantly exceeding its initial assigned share (1,905 tickets).
       - This aligns with Neha Kulkarni's operational observation that Logistics faces high resolution times and hand-offs.

    2. **Billing Queue Distortion:**
       - Initial bot routing indicated Billing was the largest queue (2,425 tickets, 20.8%).
       - Model analysis indicates that **46.3% of tickets assigned to Billing relate to shipment delivery delays or return pickup requests**.
       - True Billing workload is model-estimated at **1,601 tickets (13.8%)**.

    3. **Tier 2 Complexity Considerations:**
       - **Escalations & Warranty (Tier 2)** represents 4.8% of volume (556 tickets).
       - Per **Support Operating Policy §6**, Tier 2 cases are certified RMA investigations measured in resolution days, not weekly closed ticket volume. Leadership must consider case complexity when weighing Tier 1 vs Tier 2 staffing.

    4. **Process Fix Opportunity:**
       - Addressing intake routing errors at the chatbot level represents an estimated **₹9.21 Lakh in process-cost exposure** (3,020 mismatched tickets based on the ₹305/transfer standard in SOP §4), addressing Arjun Mehta's priority to fix processes before adding headcount.
    """)

    st.markdown("---")
    st.subheader("⚠️ Limitations & Known Issues")
    st.markdown("""
    - **Predictions Are Probabilistic:** Model outputs are predictions based on text features and domain patterns, not verified ground truth. The previous 95% heuristic metric was circular (derived from the model's own predictions) and was permanently discarded. True validation requires genuine human labeling.
    - **Validation Sample Size (30 Tickets):** The 30-ticket sample is directional and not sufficient to estimate production-wide accuracy.
    - **Multi-Intent Ambiguity:** In edge cases where customers report a hardware fault but request an immediate return/refund under the 7-day DOA policy, single-label models may capture the defect symptom rather than the refund request intent.
    - **Legacy Data Incompleteness:** Freshdesk pre-migration tickets lack transfer count telemetry.
    - **Offline Scoping:** System processes static CSV exports without live helpdesk API hooks.
    """)
