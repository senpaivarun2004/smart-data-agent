"""
Gemini-Powered Data Agent with Self-Correction Loop.

Translates natural language questions into Python/Pandas code,
executes it safely, and synthesizes human-readable insights.
"""

import os
import re
import json
from google import genai

from core.code_executor import execute_code_safely
from core.data_cleaning import get_schema_summary


# Maximum self-correction retries before giving up
MAX_RETRIES = 3


SYSTEM_PROMPT = """You are a senior data scientist assistant. You help users analyze datasets by writing Python/Pandas code.

## Rules
1. You receive the dataset schema and the user's question.
2. You MUST respond with ONLY a Python code block that answers the question.
3. The dataframe is available as `df`. Do NOT re-read or re-load any CSV files.
4. Use ONLY these libraries: `pd` (pandas), `np` (numpy), `plt` (matplotlib.pyplot), `sns` (seaborn).
5. For visualizations, you have these helper functions available:
   - `plot_distribution(df, column, title=None)` — histogram + KDE
   - `plot_bar(df, x_col, y_col, title=None, top_n=None)` — vertical bar chart
   - `plot_trend(df, date_col, metric_col, title=None)` — time-series line chart
   - `plot_top_n(df, category_col, value_col, n=10, title=None)` — horizontal bar (top N)
   - `plot_correlation_heatmap(df, title=None)` — correlation matrix
   - `plot_pie(df, category_col, value_col, title=None, top_n=8)` — donut chart
   These functions take a dataframe as the first argument and save the chart automatically.
   PREFER these helper functions over writing raw matplotlib code.
6. Store your final answer in a variable called `result`. This can be:
   - A DataFrame for tabular answers
   - A string for text answers
   - A number for single-value answers
7. Use `print()` for any intermediate output you want the user to see.
8. Always use the EXACT column names from the schema. Do NOT guess or invent column names.
9. Wrap your code in ```python ... ``` markers.
10. For date-based grouping, handle both datetime and period types properly.

## Important
- ONLY output the code block. No explanations before or after.
- Make sure the code is complete and self-contained.
- If asked to visualize, call the appropriate helper function AND compute the numeric result.
"""

INSIGHT_PROMPT = """You are a data analyst writing a report for a non-technical business manager.

Given the user's question and the analysis results below, write a clear, insightful summary in 2-4 sentences.
- Use specific numbers from the results
- Highlight key trends, outliers, or business implications
- Be conversational but professional
- Use bullet points if there are multiple findings
- Do NOT mention code, DataFrames, or technical details

Question: {question}

Results:
{results}

Write your insight summary:"""

FIX_CODE_PROMPT = """The Python code you generated failed with this error:

```
{error}
```

Here is the code that failed:
```python
{code}
```

Here is the dataset schema for reference:
{schema}

Fix the code. Remember:
- Use ONLY the exact column names from the schema
- The dataframe is available as `df`
- Store the final answer in `result`
- Wrap your code in ```python ... ``` markers
- ONLY output the fixed code block, nothing else
"""


def _extract_code(response_text: str) -> str:
    """Extract Python code from markdown code blocks in the LLM response."""
    # Try to find ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Last resort: treat the whole response as code
    return response_text.strip()


class DataAgent:
    """
    AI-powered data analysis agent using Google Gemini.

    Features:
    - Natural language to Python/Pandas code translation
    - Self-correcting execution loop (retries on errors)
    - Automatic insight synthesis
    """

    def __init__(self, api_key: str = None):
        """Initialize the Gemini client."""
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it as an environment variable or pass it directly."
            )
        self.client = genai.Client(api_key=key)
        self.model = "gemini-2.0-flash"
        self.df = None
        self.schema = None
        self.conversation_history = []

    def set_context(self, df, schema_summary: str = None):
        """Load a dataframe and its schema into agent memory."""
        self.df = df
        self.schema = schema_summary or get_schema_summary(df)
        self.conversation_history = []

    def _call_llm(self, prompt: str) -> str:
        """Make a single call to the Gemini API."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text

    def _generate_code(self, question: str) -> str:
        """Ask the LLM to generate Python code for a question."""
        prompt = f"""{SYSTEM_PROMPT}

## Dataset Schema
{self.schema}

## User Question
{question}

Generate the Python code:"""

        response_text = self._call_llm(prompt)
        return _extract_code(response_text)

    def _fix_code(self, code: str, error: str) -> str:
        """Ask the LLM to fix broken code based on the error."""
        prompt = FIX_CODE_PROMPT.format(
            error=error,
            code=code,
            schema=self.schema,
        )
        response_text = self._call_llm(prompt)
        return _extract_code(response_text)

    def _generate_insight(self, question: str, results: str) -> str:
        """Ask the LLM to synthesize a human-readable insight."""
        prompt = INSIGHT_PROMPT.format(question=question, results=results)
        return self._call_llm(prompt)

    def ask(self, question: str) -> dict:
        """
        Main entry point: answer a natural-language question about the data.

        Returns:
            dict with keys: code, result_data, chart_path, insight, attempts, error
        """
        if self.df is None:
            return {
                "code": None,
                "result_data": None,
                "chart_path": None,
                "insight": "⚠️ No dataset loaded. Please upload a CSV file first.",
                "attempts": 0,
                "error": "No dataset loaded",
            }

        # Step 1: Generate code
        code = self._generate_code(question)
        last_error = None

        # Step 2: Execute with self-correction loop
        for attempt in range(1, MAX_RETRIES + 1):
            exec_result = execute_code_safely(code, self.df)

            if exec_result["success"]:
                # Step 3: Format result data for insight generation
                result_str = ""
                if exec_result["data"] is not None:
                    import pandas as pd
                    if isinstance(exec_result["data"], pd.DataFrame):
                        result_str = exec_result["data"].to_string(max_rows=30)
                    else:
                        result_str = str(exec_result["data"])

                if exec_result["logs"]:
                    result_str += f"\n\nPrinted output:\n{exec_result['logs']}"

                if exec_result["chart"]:
                    result_str += "\n\n[A chart was generated and displayed]"

                # Step 4: Generate insight
                try:
                    insight = self._generate_insight(question, result_str)
                except Exception:
                    insight = "Analysis complete. See the results above."

                return {
                    "code": code,
                    "result_data": exec_result["data"],
                    "chart_path": exec_result["chart"],
                    "insight": insight,
                    "attempts": attempt,
                    "error": None,
                }

            # Self-correction: send error back to LLM
            last_error = exec_result["error"]
            if attempt < MAX_RETRIES:
                code = self._fix_code(code, last_error)

        # All retries exhausted
        return {
            "code": code,
            "result_data": None,
            "chart_path": None,
            "insight": f"❌ I wasn't able to answer this question after {MAX_RETRIES} attempts. "
                       f"Last error: {last_error}",
            "attempts": MAX_RETRIES,
            "error": last_error,
        }
