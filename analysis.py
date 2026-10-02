"""
analysis.py - Analytics, volume aggregation, misrouting detection,
transfer cost estimation, and validation analysis for Vireo Audio.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


INTERNAL_TRANSFER_COST_INR = 305.0  # support-policy.pdf Section 4


def load_all_data(data_dir: str = ".") -> Dict[str, pd.DataFrame]:
    """
    Loads all datasets from data_dir and performs initial preparation.
    """
    files = {
        "tickets": "tickets.csv",
        "agents": "agents.csv",
        "customers": "customers.csv",
        "orders": "orders.csv",
        "products": "products.csv",
    }
    
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Required file '{filename}' not found in '{data_dir}'")

    # Clean & enrich tickets dataframe
    raw_tickets = data["tickets"].copy()
    raw_tickets["created_at_dt"] = pd.to_datetime(raw_tickets["created_at"], errors="coerce")
    raw_tickets["first_response_at_dt"] = pd.to_datetime(raw_tickets["first_response_at"], errors="coerce")
    raw_tickets["resolved_at_dt"] = pd.to_datetime(raw_tickets["resolved_at"], errors="coerce")
    raw_tickets["month_year"] = raw_tickets["created_at_dt"].dt.strftime("%Y-%m")

    # Map agent resolving team (agent who resolved ticket)
    if "agents" in data:
        agent_map = data["agents"].drop_duplicates("agent_id").set_index("agent_id")["team"].to_dict()
        raw_tickets["resolving_team"] = raw_tickets["agent_id"].map(agent_map).fillna("Unknown")

    # Filter for primary analysis period: Jan 1, 2025 through Jun 30, 2026 inclusive
    period_mask = (raw_tickets["created_at_dt"] >= "2025-01-01") & (raw_tickets["created_at_dt"] <= "2026-06-30 23:59:59")
    tickets = raw_tickets[period_mask].copy()

    data["raw_tickets"] = raw_tickets
    data["tickets"] = tickets
    return data


def get_dataset_statistics(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Computes summary statistics across all provided datasets.
    """
    tickets = data["tickets"]
    agents = data.get("agents", pd.DataFrame())
    customers = data.get("customers", pd.DataFrame())
    orders = data.get("orders", pd.DataFrame())
    products = data.get("products", pd.DataFrame())

    valid_csat = tickets["csat_score"].dropna()
    avg_csat = float(valid_csat.mean()) if len(valid_csat) > 0 else 0.0

    # Source systems and transfers notes (legacy transfers are blank, not 0)
    helpdesk_tickets = tickets[tickets["source_system"] == "helpdesk"]
    legacy_tickets = tickets[tickets["source_system"] == "legacy_fd"]

    return {
        "total_tickets": int(len(tickets)),
        "total_customers": int(len(customers)),
        "total_agents": int(agents["agent_id"].nunique()) if not agents.empty else 0,
        "total_orders": int(len(orders)),
        "total_products": int(len(products)),
        "date_range": (
            str(tickets["created_at_dt"].min().date()),
            str(tickets["created_at_dt"].max().date())
        ) if not tickets.empty else ("N/A", "N/A"),
        "channel_counts": tickets["channel"].value_counts().to_dict(),
        "status_counts": tickets["status"].value_counts().to_dict(),
        "priority_counts": tickets["priority"].value_counts().to_dict(),
        "source_system_counts": tickets["source_system"].value_counts().to_dict(),
        "helpdesk_ticket_count": int(len(helpdesk_tickets)),
        "legacy_ticket_count": int(len(legacy_tickets)),
        "avg_csat": round(avg_csat, 2),
        "csat_response_rate": round(float(len(valid_csat) / len(tickets)), 4) if len(tickets) > 0 else 0.0,
    }


def compute_monthly_volume(
    tickets_df: pd.DataFrame,
    group_col: str = "category"
) -> pd.DataFrame:
    """
    Computes monthly ticket volume broken down by a specific column (category or team).
    """
    if group_col not in tickets_df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")

    monthly = (
        tickets_df.groupby(["month_year", group_col])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    return monthly


def analyze_misrouting(classified_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes category and team routing differences between original intake tags and model predictions.
    """
    df = classified_df.copy()
    
    # Check flags
    df["category_mismatch"] = df["category"] != df["predicted_category"]
    df["team_mismatch"] = df["assigned_team"] != df["predicted_team"]

    total_tickets = len(df)
    category_misrouted_count = int(df["category_mismatch"].sum())
    team_misrouted_count = int(df["team_mismatch"].sum())

    # Breakdown by original assigned team
    team_breakdown = (
        df.groupby("assigned_team")
        .agg(
            total_assigned=("ticket_id", "count"),
            mismatch_count=("team_mismatch", "sum"),
            mismatch_rate=("team_mismatch", "mean")
        )
        .reset_index()
    )
    team_breakdown["mismatch_rate"] = team_breakdown["mismatch_rate"].round(4)

    # Breakdown of what Billing tickets actually classify as
    billing_predicted_cats = (
        df[df["assigned_team"] == "Billing"]["predicted_category"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_tickets": total_tickets,
        "category_misrouted_count": category_misrouted_count,
        "category_misrouting_rate": round(category_misrouted_count / total_tickets, 4) if total_tickets > 0 else 0.0,
        "team_misrouted_count": team_misrouted_count,
        "team_misrouting_rate": round(team_misrouted_count / total_tickets, 4) if total_tickets > 0 else 0.0,
        "team_breakdown": team_breakdown,
        "billing_predicted_categories": billing_predicted_cats,
    }


def calculate_transfer_costs(
    classified_df: pd.DataFrame,
    transfer_cost_inr: float = INTERNAL_TRANSFER_COST_INR
) -> Dict[str, Any]:
    """
    Calculates internal transfer costs and estimated process-cost exposure
    using the Rs 305/transfer SOP standard (Support Policy Section 4).
    """
    df = classified_df.copy()

    # Recorded transfers in Helpdesk system (legacy tickets are blank / NaN per SOP §9)
    recorded_transfers = float(df["transfers"].dropna().sum())
    recorded_transfer_cost = recorded_transfers * transfer_cost_inr

    # Team mismatch count (potential misrouting process-cost exposure)
    mismatch_count = int((df["assigned_team"] != df["predicted_team"]).sum())
    estimated_mismatch_exposure = mismatch_count * transfer_cost_inr

    return {
        "unit_transfer_cost_inr": transfer_cost_inr,
        "recorded_transfers_count": int(recorded_transfers),
        "recorded_transfer_cost_inr": round(recorded_transfer_cost, 2),
        "mismatched_tickets_count": mismatch_count,
        "estimated_mismatch_exposure_inr": round(estimated_mismatch_exposure, 2),
    }


def analyze_validation_sample(sample_path: str = "manual_validation_reviews.csv") -> Dict[str, Any]:
    """
    Loads and analyzes the genuine manual validation sample.
    Calculates agreement and error metrics ONLY based on completed human reviews.
    Suppresses accuracy claims if no reviews have been submitted.
    """
    if not os.path.exists(sample_path):
        return {"available": False, "status": "missing"}

    sample_df = pd.read_csv(sample_path)
    total_samples = len(sample_df)

    # Check for completed human reviews
    cat_series = sample_df["human_validated_category"].fillna("").astype(str).str.strip()
    reviewed_mask = (cat_series != "") & (cat_series.str.lower() != "nan")
    reviewed_df = sample_df[reviewed_mask].copy()
    reviewed_count = int(len(reviewed_df))
    remaining_count = int(total_samples - reviewed_count)

    if reviewed_count == 0:
        return {
            "available": True,
            "status": "pending",
            "message": "Manual validation pending — no accuracy claim available.",
            "total_samples": total_samples,
            "reviewed_count": 0,
            "remaining_count": remaining_count,
            "model_category_agreement_count": 0,
            "model_category_agreement_rate": None,
            "model_team_agreement_count": 0,
            "model_team_agreement_rate": None,
            "category_error_rate": None,
            "error_breakdown": {},
            "sample_df": sample_df,
            "reviewed_df": reviewed_df,
        }

    # Calculate actual agreement based strictly on completed human reviews
    cat_matches = int((reviewed_df["ai_predicted_category"] == reviewed_df["human_validated_category"]).sum())
    cat_agreement_rate = round(cat_matches / reviewed_count, 4)
    cat_error_rate = round(1.0 - cat_agreement_rate, 4)

    team_matches = int((reviewed_df["ai_predicted_team"] == reviewed_df["human_validated_team"]).sum())
    team_agreement_rate = round(team_matches / reviewed_count, 4)

    # Bot agreement with human validator
    bot_matches = int((reviewed_df["original_category"] == reviewed_df["human_validated_category"]).sum())
    bot_agreement_rate = round(bot_matches / reviewed_count, 4)

    # Error breakdown
    mismatches = reviewed_df[reviewed_df["ai_predicted_category"] != reviewed_df["human_validated_category"]]
    error_breakdown = {}
    for _, r in mismatches.iterrows():
        err_key = f"AI predicted '{r['ai_predicted_category']}' vs Human '{r['human_validated_category']}'"
        error_breakdown[err_key] = error_breakdown.get(err_key, 0) + 1

    return {
        "available": True,
        "status": "in_progress" if reviewed_count < total_samples else "completed",
        "message": f"Validation based on {reviewed_count} of {total_samples} human reviews.",
        "total_samples": total_samples,
        "reviewed_count": reviewed_count,
        "remaining_count": remaining_count,
        "model_category_agreement_count": cat_matches,
        "model_category_agreement_rate": cat_agreement_rate,
        "model_team_agreement_count": team_matches,
        "model_team_agreement_rate": team_agreement_rate,
        "bot_agreement_count": bot_matches,
        "bot_agreement_rate": bot_agreement_rate,
        "category_error_rate": cat_error_rate,
        "error_breakdown": error_breakdown,
        "sample_df": sample_df,
        "reviewed_df": reviewed_df,
    }


def get_workload_comparison(classified_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a clear workload comparison table between original assigned volume
    and model-predicted workload, noting Tier 1 vs Tier 2 operational differences.
    """
    orig_teams = classified_df["assigned_team"].value_counts()
    pred_teams = classified_df["predicted_team"].value_counts()

    comparison_df = pd.DataFrame({
        "Original Assigned Volume": orig_teams,
        "Model-Predicted Workload": pred_teams
    }).fillna(0).astype(int)

    comparison_df["Volume Difference"] = comparison_df["Model-Predicted Workload"] - comparison_df["Original Assigned Volume"]
    comparison_df["Share of Model Workload (%)"] = (
        (comparison_df["Model-Predicted Workload"] / len(classified_df)) * 100
    ).round(2)

    # Note Tier designation based on Support Policy Section 6
    tier_map = {
        "Chat Frontline": "Tier 1",
        "Email Frontline": "Tier 1",
        "Voice Frontline": "Tier 1",
        "Logistics": "Tier 1 / Specialized",
        "Billing": "Tier 1 / Specialized",
        "Returns Desk": "Tier 1 / Specialized",
        "Escalations & Warranty": "Tier 2 (Certified RMA)"
    }
    comparison_df["Tier Level"] = comparison_df.index.map(tier_map)

    comparison_df = comparison_df.sort_values(by="Model-Predicted Workload", ascending=False)

    return {
        "comparison_table": comparison_df.reset_index().rename(columns={"index": "Support Team"}),
        "logistics_share": comparison_df.loc["Logistics", "Share of Model Workload (%)"] if "Logistics" in comparison_df.index else 0.0,
        "billing_share": comparison_df.loc["Billing", "Share of Model Workload (%)"] if "Billing" in comparison_df.index else 0.0,
    }
