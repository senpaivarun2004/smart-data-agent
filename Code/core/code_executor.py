"""
Safe Code Execution Sandbox.

Runs LLM-generated Python code in a restricted namespace,
capturing output, results, and chart paths.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from core import visualizations as viz


def execute_code_safely(code_str: str, df: pd.DataFrame) -> dict:
    """
    Execute LLM-generated Python code in a sandboxed namespace.

    The code has access to:
        - df: the user's DataFrame
        - pd, np, plt, sns: standard data science libraries
        - viz: the visualization helper module (plot_distribution, plot_bar, etc.)

    Returns:
        dict with keys: success, data, logs, chart, error
    """
    # Clean up any previous chart
    chart_path = "outputs/temp_chart.png"
    if os.path.exists(chart_path):
        os.remove(chart_path)

    # Build the restricted execution namespace
    exec_namespace = {
        # Data
        "df": df.copy(),
        # Libraries
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        # Visualization helpers
        "viz": viz,
        "plot_distribution": viz.plot_distribution,
        "plot_bar": viz.plot_bar,
        "plot_trend": viz.plot_trend,
        "plot_top_n": viz.plot_top_n,
        "plot_correlation_heatmap": viz.plot_correlation_heatmap,
        "plot_pie": viz.plot_pie,
        # Built-ins (limited)
        "print": print,
        "len": len,
        "range": range,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "round": round,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "enumerate": enumerate,
        "zip": zip,
        "type": type,
        "isinstance": isinstance,
        "True": True,
        "False": False,
        "None": None,
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        # Execute the code
        exec(code_str, {"__builtins__": {}}, exec_namespace)

        sys.stdout = old_stdout
        captured_logs = captured_output.getvalue()

        # Try to extract a 'result' variable if the code set one
        returned_data = exec_namespace.get("result", None)

        # If no explicit result, check if there's a new DataFrame
        if returned_data is None:
            for key, val in exec_namespace.items():
                if key not in ("df", "pd", "np", "plt", "sns", "viz") and isinstance(val, pd.DataFrame):
                    returned_data = val
                    break

        # Check if code generated a chart
        has_chart = os.path.exists(chart_path)

        return {
            "success": True,
            "data": returned_data,
            "logs": captured_logs,
            "chart": chart_path if has_chart else None,
            "error": None,
        }

    except Exception as execution_error:
        sys.stdout = old_stdout
        return {
            "success": False,
            "data": None,
            "logs": None,
            "chart": None,
            "error": str(execution_error),
        }
