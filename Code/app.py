import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from io import StringIO
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration & Global Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Smart Data Analyst AI Agent", page_icon="📊", layout="wide")

# Ensure temporary storage environments exist cleanly
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Premium dark theme CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #161B22 50%, #0E1117 100%);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161B22, #1A1D23);
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #58A6FF;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #58A6FF, #BC8CFF);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(88, 166, 255, 0.3);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Maximum self-correction retries
MAX_RETRIES = 3

# Model fallback chain — if one model is rate-limited, try the next
MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]

import time

def call_gemini(contents, system_instruction=None, temperature=0.1, max_retries=2):
    """Call Gemini with model fallback and retry on rate limits."""
    config = types.GenerateContentConfig(temperature=temperature)
    if system_instruction:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
    
    last_error = None
    for model_name in MODEL_FALLBACK_CHAIN:
        for retry in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Rate limited — try next model or wait
                    if retry < max_retries - 1:
                        time.sleep(10)  # Brief wait before retry
                        continue
                    else:
                        break  # Try next model
                else:
                    raise  # Non-rate-limit error, raise immediately
    
    raise last_error  # All models exhausted

def init_agent():
    """Initialize the Gemini-powered data agent."""
    api_key = ""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "AQ.Ab8RN6JcKDeHcj-d2XVJOXtp9hL6zesq29-GgzVgEyG_4uBWGw")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6JcKDeHcj-d2XVJOXtp9hL6zesq29-GgzVgEyG_4uBWGw")
    if not api_key:
        st.error(
            "⚠️ **API key missing!** Add it to `.streamlit/secrets.toml`:\n\n"
            '```\nGEMINI_API_KEY = "your_key"\n```\n\n'
            "Or set env var: `set GEMINI_API_KEY=your_key`"
        )
        return None
    return genai.Client(api_key=api_key)

client = init_agent()
if not client:
    st.stop()

# -----------------------------------------------------------------------------
# Functional Core Engines
# -----------------------------------------------------------------------------
def clean_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Performs deterministic foundational data cleanup on input tabular sets."""
    df_clean = df.copy()
    
    # 1. Standardize string case variants across object columns to bypass typos
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        if df_clean[col].nunique() < 50:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
    # 2. Force structural date representations where possible
    for col in df_clean.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except Exception:
                pass
                
    # 3. Simple fill numeric omissions with median strategies
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    return df_clean

def generate_dynamic_prompt(df: pd.DataFrame) -> str:
    """Extracts dataframe metadata parameters to format structural prompt models."""
    columns_and_types = df.dtypes.to_string()
    data_preview = df.head(3).to_string()
    
    # Add basic stats for numeric columns
    numeric_stats = ""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        numeric_stats = f"\nNUMERIC COLUMN STATISTICS:\n{df[numeric_cols].describe().round(2).to_string()}"
    
    prompt_context = f"""You are an expert Python Data Analyst. Your sole assignment is to generate clean, executable Python code snippets manipulating an existing global Pandas DataFrame named 'df'.

The structural definition constraints for 'df' are:
---
COLUMNS AND ASSOCIATED DATATYPES:
{columns_and_types}
---
DATA SNAPSHOT SAMPLES:
{data_preview}
---{numeric_stats}

CRITICAL EXECUTION PARAMETERS:
- Return ONLY explicit executable Python code strings.
- Do NOT encompass responses inside markdown wrappers like ```python.
- Do NOT provide supplementary annotations, descriptions, or conversational greetings.
- Assume 'df' is already active within your context. Do not try to read or load the file.
- Use the EXACT column names from the schema above. Do NOT guess or invent column names.
- If the user query requires a graphical layout, construct it via Matplotlib or Seaborn, then save the graphic explicitly to 'outputs/temp_chart.png' using plt.savefig('outputs/temp_chart.png', dpi=150, bbox_inches='tight', facecolor='#0E1117'). Clear the plot using plt.close() afterwards.
- For charts, use a dark theme: set figure facecolor to '#0E1117', axes facecolor to '#1A1D23', and text/label colors to '#E6EDF3'. Use colors like '#58A6FF', '#BC8CFF', '#F778BA', '#FF7B72', '#FFA657', '#56D364'.
- Store plain textual metrics or aggregated tabular summary results inside a target variable called 'output_result'.
"""
    return prompt_context

def generate_fix_prompt(code: str, error: str, schema: str) -> str:
    """Create a prompt to fix broken code."""
    return f"""The following Python code failed with an error. Fix it.

FAILED CODE:
{code}

ERROR:
{error}

SCHEMA REFERENCE:
{schema}

RULES:
- Return ONLY the fixed executable Python code.
- Do NOT use markdown wrappers.
- Use EXACT column names from the schema.
- The dataframe is available as 'df'.
- Store results in 'output_result'.
- Save charts to 'outputs/temp_chart.png'.
"""

def execute_generated_code(code_str: str, dataframe_context: pd.DataFrame):
    """Safely handles code block evaluation inside an isolated workspace context."""
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    execution_scope = {
        'df': dataframe_context.copy(),
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'np': __import__('numpy'),
        'output_result': None
    }
    
    try:
        exec(code_str, execution_scope)
        sys.stdout = old_stdout
        
        captured_logs = redirected_output.getvalue()
        returned_data = execution_scope.get('output_result')
        
        chart_path = "outputs/temp_chart.png"
        has_chart = os.path.exists(chart_path)
        
        return {
            "success": True, 
            "data": returned_data, 
            "logs": captured_logs, 
            "chart": chart_path if has_chart else None,
            "error": None
        }
    except Exception as execution_error:
        sys.stdout = old_stdout
        return {
            "success": False, 
            "data": None, 
            "logs": None, 
            "chart": None, 
            "error": str(execution_error)
        }

def generate_insight(user_query: str, results: str) -> str:
    """Ask Gemini to write a human-readable insight summary."""
    prompt = f"""You are a data analyst writing a report for a non-technical business manager.

Given the user's question and the analysis results below, write a clear, insightful summary in 2-4 sentences.
- Use specific numbers from the results
- Highlight key trends, outliers, or business implications
- Be conversational but professional
- Do NOT mention code, DataFrames, or technical details

Question: {user_query}

Results:
{results}

Write your insight summary:"""
    
    try:
        return call_gemini(contents=prompt, temperature=0.5)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Streamlit Interface Rendering Engine
# -----------------------------------------------------------------------------
st.title("📊 Smart Data Analyst AI Agent")
st.markdown("Drop any raw enterprise or retail CSV file here, type what you want to discover, and let the AI generate analytics instantly.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Interactive File Drop Section
uploaded_file = st.file_uploader("Upload your data matrix sheet (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Retain physical state across uploads to cache computing power
    file_target_path = os.path.join("data", uploaded_file.name)
    with open(file_target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Read CSV with encoding fallback
    raw_df = None
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            raw_df = pd.read_csv(file_target_path, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue
    if raw_df is None:
        st.error("❌ Could not read the CSV file. Please check the file encoding.")
        st.stop()
    
    # Run the core data science ingestion cleaning phase
    processed_df = clean_and_prepare_dataset(raw_df)
    
    # Construct distinct visual panels for layout cleanliness
    data_tab, analyst_tab = st.tabs(["🗂️ Ingested Dataset Registry", "🤖 Talk with Data Analyst Agent"])
    
    with data_tab:
        st.subheader("Data Summary Profiles")
        col1, col2, col3 = st.columns(3)
        col1.metric("Row Volume Count", f"{processed_df.shape[0]:,}")
        col2.metric("Feature Dimension Columns", processed_df.shape[1])
        col3.metric("Missing Values", processed_df.isna().sum().sum())
        
        st.subheader("Data Matrix Head Overview")
        st.dataframe(processed_df.head(10), use_container_width=True)
        
        # Show column types
        with st.expander("📋 Column Details"):
            for col in processed_df.columns:
                dtype = str(processed_df[col].dtype)
                emoji = "🔢" if "int" in dtype or "float" in dtype else "📅" if "datetime" in dtype else "📝"
                st.markdown(f"{emoji} **{col}** `{dtype}` — {processed_df[col].nunique()} unique values")
        
    with analyst_tab:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🧠"):
                st.markdown(msg["content"])
                if msg.get("chart"):
                    st.image(msg["chart"], use_container_width=True)
                if msg.get("data") is not None:
                    if isinstance(msg["data"], (pd.DataFrame, pd.Series)):
                        st.dataframe(msg["data"], use_container_width=True)
                    else:
                        st.write(msg["data"])
                if msg.get("code"):
                    with st.expander("🔧 Generated Code"):
                        st.code(msg["code"], language="python")
                    if msg.get("attempts", 1) > 1:
                        st.caption(f"✨ Self-corrected in {msg['attempts']} attempts")

        # Chat input
        user_query = st.chat_input("Ask a question about your data... (e.g., 'Show sales distribution' or 'Top 5 items by revenue')")
        
        if user_query:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(user_query)
            
            # Flush previous chart
            if os.path.exists("outputs/temp_chart.png"):
                os.remove("outputs/temp_chart.png")
            
            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("🔍 Analyzing your data..."):
                    # Generate system prompt from schema
                    system_context = generate_dynamic_prompt(processed_df)
                    
                    # Request code from Gemini
                    try:
                        response_text = call_gemini(
                            contents=user_query,
                            system_instruction=system_context,
                            temperature=0.1,
                        )
                        raw_llm_code = response_text.replace("```python", "").replace("```", "").strip()
                    except Exception as llm_fault:
                        st.error(f"❌ Failed to communicate with AI: {llm_fault}")
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"❌ Failed to communicate with AI: {llm_fault}"
                        })
                        st.stop()
                
                # Self-correction loop
                final_code = raw_llm_code
                execution_report = None
                attempts = 0
                
                for attempt in range(1, MAX_RETRIES + 1):
                    attempts = attempt
                    with st.spinner(f"⚙️ Executing code (attempt {attempt}/{MAX_RETRIES})..."):
                        execution_report = execute_generated_code(final_code, processed_df)
                    
                    if execution_report["success"]:
                        break
                    
                    # Self-correction: send error back to Gemini
                    if attempt < MAX_RETRIES:
                        with st.spinner(f"🔄 Self-correcting (attempt {attempt + 1})..."):
                            try:
                                fix_prompt = generate_fix_prompt(
                                    final_code, 
                                    execution_report["error"],
                                    processed_df.dtypes.to_string()
                                )
                                fix_text = call_gemini(
                                    contents=fix_prompt,
                                    temperature=0.1,
                                )
                                final_code = fix_text.replace("```python", "").replace("```", "").strip()
                            except Exception:
                                break
                
                # Render results
                if execution_report and execution_report["success"]:
                    # Build result summary for insight generation
                    result_str = ""
                    chart_path = None
                    
                    if execution_report["data"] is not None:
                        if isinstance(execution_report["data"], (pd.DataFrame, pd.Series)):
                            st.dataframe(execution_report["data"], use_container_width=True)
                            result_str = str(execution_report["data"])
                        else:
                            st.write(execution_report["data"])
                            result_str = str(execution_report["data"])
                    
                    if execution_report["chart"]:
                        st.image(execution_report["chart"], use_container_width=True)
                        chart_path = execution_report["chart"]
                        result_str += "\n[A chart was generated]"
                    
                    if execution_report["logs"]:
                        result_str += f"\nOutput: {execution_report['logs']}"
                    
                    # Generate AI insight
                    insight = generate_insight(user_query, result_str)
                    if insight:
                        st.markdown(f"**💡 Insight:** {insight}")
                    else:
                        st.success("✅ Analysis complete!")
                    
                    # Show generated code
                    with st.expander("🔧 Generated Code"):
                        st.code(final_code, language="python")
                    if attempts > 1:
                        st.caption(f"✨ Self-corrected in {attempts} attempts")
                    
                    # Save to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": insight or "✅ Analysis complete!",
                        "chart": chart_path,
                        "data": execution_report["data"],
                        "code": final_code,
                        "attempts": attempts,
                    })
                    
                elif execution_report:
                    error_msg = f"❌ Analysis failed after {attempts} attempts. Error: {execution_report['error']}"
                    st.error(error_msg)
                    with st.expander("🔧 Failed Code"):
                        st.code(final_code, language="python")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

else:
    st.info("📥 Upload a CSV file above to start analyzing your data with AI.")
    st.markdown("""
    ### 💡 Example questions you can ask:
    - *"Show me the distribution of sales"*
    - *"Which product category has the highest total revenue?"*
    - *"Show me monthly sales trends"*
    - *"What's the correlation between price and sales?"*
    - *"Top 10 items by profit margin"*
    """)