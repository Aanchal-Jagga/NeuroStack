import pandas as pd
import io
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="C:/NeuroStack/.env")

from tavily import TavilyClient
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

AgentState = Dict[str, Any]

# # ✅ Initialize Tavily client (API key must be set in env vars)
# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def analyze_data_node(state: AgentState) -> AgentState:
    """Analyze uploaded CSV and generate report + plots"""
    try:
        state["csv_file"].seek(0)
        df = pd.read_csv(state["csv_file"], encoding='utf-8-sig')

        if df.empty:
            state["report"] = "⚠️ CSV file is empty. Please provide valid financial data."
            state["content"] = []
            return state

        # ----------------------------
        # Basic analysis
        # ----------------------------
        summary_lines = []
        summary_lines.append("📊 Financial Summary\n")
        summary_lines.append(f"Number of rows: {len(df)}")
        summary_lines.append(f"Columns: {', '.join(df.columns)}\n")

        # Compute simple statistics
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            summary_lines.append("Summary statistics:\n")
            summary_lines.append(df[numeric_cols].describe().to_string())
        else:
            summary_lines.append("No numeric columns found for analysis.")

        # ----------------------------
        # Competitor Analysis (Tavily)
        # ----------------------------
        if state.get("competitors"):
            summary_lines.append("\nCompetitor Analysis:")
            for comp in state["competitors"]:
                try:
                    query = f"Latest financial performance 2024-2025 of {comp}"
                    response = tavily.search(query=query, max_results=3)
                    snippets = [r["content"] for r in response["results"]]

                    if snippets:
                        summary_lines.append(f"- {comp}:")
                        for s in snippets:
                            summary_lines.append(f"    • {s}")
                    else:
                        summary_lines.append(f"- {comp}: No relevant data found.")

                except Exception as e:
                    summary_lines.append(f"- {comp}: Tavily fetch error ({e})")

        state["report"] = "\n".join(summary_lines)

        # ----------------------------
        # Generate plots
        # ----------------------------
        plots = []
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            df[col].plot(kind='line', title=f"{col} Trend", grid=True)
            plt.xlabel("Index")
            plt.ylabel(col)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            # ✅ Store as tuple (name, buffer)
            plots.append((col, buf))

        state["content"] = plots
        state["current_step"].append("analyze_data_node ✅")

        return state

    except Exception as e:
        state["report"] = f"❌ Error analyzing CSV: {e}"
        state["content"] = []
        return state


def run_financial_report(state: AgentState) -> AgentState:
    """Run the full financial report pipeline"""
    state["current_step"] = []

    # Step 1: Analyze CSV
    state = analyze_data_node(state)
    state["current_step"].append("run_financial_report ✅")

    return state
