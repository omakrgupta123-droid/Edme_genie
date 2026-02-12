import streamlit as st
import os
import logging
import json
import datetime
import time
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from databricks.sdk import WorkspaceClient
import boto3
from langchain_aws import ChatBedrock

# -----------------------
# Load env
# -----------------------
load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")

WORKSPACE_CLIENT = WorkspaceClient(
    host=DATABRICKS_HOST,
    token=DATABRICKS_TOKEN,
)

# -----------------------
# Display & Performance Settings
# -----------------------
DISPLAY_TABLE = False
ENABLE_CSV_DOWNLOAD = False
SHOW_SUMMARY_HEADER = False

# -----------------------
# Bedrock / LangChain Config
# -----------------------
ENABLE_BEDROCK_CHARTS = True
BEDROCK_PROFILE = "qms-assumed-role"
BEDROCK_REGION = "ap-south-1"
BEDROCK_MODEL_ID = "global.anthropic.claude-opus-4-5-20251101-v1:0"

# Keywords for summary-style requests
SUMMARY_KEYWORDS = ("summary", "overview", "profile", "describe", "stats", "statistics")
CHART_KEYWORDS = ("chart", "graph", "plot", "visual", "visualize", "trend", "distribution", "compare", "correlation", "insight", "display")
TABLE_KEYWORDS = ("table", "tabular", "dataframe", "rows", "columns", "show data", "show table")

def is_summary_request(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in SUMMARY_KEYWORDS)

def is_chart_request(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in CHART_KEYWORDS)

def is_table_request(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in TABLE_KEYWORDS)

def _format_value(value):
    if pd.isna(value):
        return "NULL"
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return str(value)
        try:
            return f"{value:,.4g}" if isinstance(value, float) else f"{value:,}"
        except Exception:
            return str(value)
    return str(value)

def _json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (datetime.date, datetime.datetime)):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value

def summarize_dataframe(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No rows returned."
    if len(df) == 1:
        row = df.iloc[0].to_dict()
        parts = [f"{k} = {_format_value(v)}" for k, v in row.items()]
        return "Summary: " + ", ".join(parts) + "."
    cols = ", ".join(list(df.columns)[:8])
    if len(df.columns) > 8:
        cols += ", ..."
    return f"Summary: results returned for columns {cols}."

def summarize_result(schema, rows) -> str:
    if not rows:
        return "No rows returned."
    if len(rows) == 1 and schema:
        row = rows[0]
        parts = []
        for idx, col in enumerate(schema):
            if isinstance(col, dict):
                name = col.get("name") or f"col_{idx}"
            else:
                name = str(col)
            value = row[idx] if idx < len(row) else None
            parts.append(f"{name} = {_format_value(value)}")
        return "Summary: " + ", ".join(parts) + "."
    col_names = []
    if schema:
        for col in schema[:8]:
            if isinstance(col, dict):
                col_names.append(col.get("name") or "")
            else:
                col_names.append(str(col))
    cols = ", ".join([c for c in col_names if c])
    if schema and len(schema) > 8:
        cols += ", ..."
    if cols:
        return f"Summary: results returned for columns {cols}."
    return "Summary: results returned successfully."

@st.cache_resource(show_spinner=False)
def get_bedrock_llm():
    if not ENABLE_BEDROCK_CHARTS:
        return None
    try:
        session = boto3.Session(profile_name=BEDROCK_PROFILE, region_name=BEDROCK_REGION)
        client = session.client("bedrock-runtime", region_name=BEDROCK_REGION)
        return ChatBedrock(model_id=BEDROCK_MODEL_ID, client=client)
    except Exception as e:
        logging.warning(f"Bedrock client init failed: {e}")
        return None

def _extract_json(text: str):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None

def _validate_genie_viz_spec(spec: dict):
    if not isinstance(spec, dict):
        return None
    chart_type = (spec.get("chart_type") or "").lower()
    if chart_type not in ("pie", "bar", "line"):
        return None
    category_col = spec.get("category_col")
    aggregation = (spec.get("aggregation") or "count").lower()
    if aggregation not in ("count", "sum", "mean"):
        aggregation = "count"
    title = spec.get("title") or "Chart"
    value_col = spec.get("value_col")
    return {
        "chart_type": chart_type,
        "category_col": category_col,
        "value_col": value_col,
        "aggregation": aggregation,
        "title": title,
    }

def extract_genie_viz_spec(attachments, answer_text: str):
    candidates = []
    for att in attachments or []:
        text = (att.get("text") or {}).get("content")
        if text:
            candidates.append(text)
    if answer_text:
        candidates.append(answer_text)
    for text in candidates:
        spec = _extract_json(text)
        spec = _validate_genie_viz_spec(spec)
        if spec:
            return spec
    return None

@st.cache_data(show_spinner=False)
def infer_chart_spec(context, question: str, answer_text: str, genie_spec: dict | None):
    llm = get_bedrock_llm()
    if not llm:
        return None
    genie_block = json.dumps(genie_spec) if genie_spec else "null"
    prompt = f"""
You are a data visualization assistant. Choose the best chart to visualize the data.
Return JSON only (no extra text) with keys:
chart_type: one of ["pie","bar","line"]
category_col: a categorical column name (string)
value_col: a numeric column name or null
aggregation: one of ["count","sum","mean"]
title: short chart title

Guidelines:
- Use pie for category distributions (counts).
- Use bar for category + numeric values.
- Use line for time trends or ordered numeric series.
- If value_col is null, use count.
- Choose only columns that exist.

User question: {question}
Genie answer: {answer_text}
Genie structured spec (use this as primary guidance): {genie_block}
Context: {json.dumps(context)}
"""
    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return _extract_json(text)
    except Exception as e:
        logging.warning(f"Bedrock chart spec failed: {e}")
        return None

def _limit_categories(chart_df: pd.DataFrame, cat_col: str, value_col: str, top_n: int = 10):
    if chart_df is None or chart_df.empty:
        return chart_df
    if len(chart_df) <= top_n:
        return chart_df
    sorted_df = chart_df.sort_values(value_col, ascending=False)
    top_df = sorted_df.head(top_n).copy()
    other_sum = sorted_df[value_col].iloc[top_n:].sum()
    other_row = pd.DataFrame({cat_col: ["Other"], value_col: [other_sum]})
    return pd.concat([top_df, other_row], ignore_index=True)

def build_chart_context(df: pd.DataFrame):
    sample_rows = []
    if df is not None and not df.empty:
        for _, row in df.head(50).iterrows():
            sample_rows.append({k: _json_safe(v) for k, v in row.to_dict().items()})
    dtypes = {c: str(df[c].dtype) for c in df.columns} if df is not None else {}
    uniques = {}
    if df is not None and not df.empty:
        for c in df.columns:
            try:
                uniques[c] = int(df[c].nunique(dropna=False))
            except Exception:
                uniques[c] = None
    return {
        "columns": list(df.columns) if df is not None else [],
        "dtypes": dtypes,
        "unique_counts": uniques,
        "sample": sample_rows,
    }

def build_chart_payload(df: pd.DataFrame, question: str, answer_text: str, genie_spec: dict | None):
    if df is None or df.empty:
        return None
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    chart_spec = None
    if ENABLE_BEDROCK_CHARTS:
        context = build_chart_context(df)
        chart_spec = infer_chart_spec(context, question, answer_text, genie_spec)
    elif genie_spec:
        chart_spec = genie_spec

    if chart_spec:
        chart_type = (chart_spec.get("chart_type") or "").lower()
        cat_col = chart_spec.get("category_col")
        val_col = chart_spec.get("value_col")
        agg = (chart_spec.get("aggregation") or "count").lower()
        title = chart_spec.get("title") or "Chart"

        if cat_col in df.columns:
            if val_col in df.columns and pd.api.types.is_numeric_dtype(df[val_col]):
                if agg == "mean":
                    chart_df = df.groupby(cat_col, dropna=False)[val_col].mean().reset_index()
                elif agg == "sum":
                    chart_df = df.groupby(cat_col, dropna=False)[val_col].sum().reset_index()
                else:
                    chart_df = df.groupby(cat_col, dropna=False)[val_col].count().reset_index()
                value_col = val_col
            else:
                chart_df = df.groupby(cat_col, dropna=False).size().reset_index(name="count")
                value_col = "count"

            chart_df = _limit_categories(chart_df, cat_col, value_col, top_n=10)
            return {
                "chart_type": chart_type if chart_type in ("pie", "bar", "line") else "bar",
                "x": cat_col,
                "y": value_col,
                "title": title,
                "data": chart_df.to_dict(orient="records"),
            }

    # Fallback heuristic
    if len(numeric_cols) >= 1 and len(non_numeric_cols) >= 1:
        cat = non_numeric_cols[0]
        num = numeric_cols[0]
        chart_df = df[[cat, num]].copy()
        chart_df = chart_df.groupby(cat, dropna=False)[num].sum().reset_index()
        chart_df = _limit_categories(chart_df, cat, num, top_n=10)
        return {
            "chart_type": "bar",
            "x": cat,
            "y": num,
            "title": f"{num} by {cat}",
            "data": chart_df.to_dict(orient="records"),
        }
    if len(non_numeric_cols) >= 1:
        cat = non_numeric_cols[0]
        chart_df = df[[cat]].copy()
        chart_df = chart_df.value_counts(dropna=False).reset_index(name="count")
        chart_df = _limit_categories(chart_df, cat, "count", top_n=10)
        return {
            "chart_type": "pie",
            "x": cat,
            "y": "count",
            "title": f"{cat} distribution",
            "data": chart_df.to_dict(orient="records"),
        }
    if len(numeric_cols) >= 1:
        chart_df = df[numeric_cols].copy()
        return {
            "chart_type": "line",
            "x": None,
            "y": None,
            "title": "Trend",
            "data": chart_df.to_dict(orient="records"),
            "series": numeric_cols,
        }
    return None

# -----------------------
# Logging Setup
# -----------------------
LOG_FILE = "genie_app.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("🚀 Genie Streamlit app started (FORCED SQL MODE)")

# -----------------------
# Session State
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "result_cache" not in st.session_state:
    st.session_state.result_cache = {}
if "question_cache" not in st.session_state:
    st.session_state.question_cache = {}

# -----------------------
# Genie API
# -----------------------

def start_conversation(question: str):
    forced_prompt = f"""
User question:
{question}
"""

    logging.info(f"QUESTION: {question}")
    msg = WORKSPACE_CLIENT.genie.start_conversation_and_wait(
        space_id=GENIE_SPACE_ID,
        content=forced_prompt,
    )
    data = msg.as_dict() if hasattr(msg, "as_dict") else msg

    logging.info(f"Conversation ID: {data.get('conversation_id')}")
    logging.info(f"Message ID: {data.get('message_id')}")

    return data


def get_query_result(conversation_id, message_id, attachment_id):
    result = WORKSPACE_CLIENT.genie.get_message_attachment_query_result(
        space_id=GENIE_SPACE_ID,
        conversation_id=conversation_id,
        message_id=message_id,
        attachment_id=attachment_id,
    )
    return result.as_dict() if hasattr(result, "as_dict") else result

def get_query_result_with_retry(conversation_id, message_id, attachment_id, max_attempts: int = 6, base_delay: float = 0.7):
    last_result = None
    for attempt in range(max_attempts):
        result = get_query_result(conversation_id, message_id, attachment_id)
        last_result = result
        statement = (result.get("statement_response") or {}) if isinstance(result, dict) else {}
        status = statement.get("status") if isinstance(statement, dict) else None
        state = None
        if isinstance(status, dict):
            state = status.get("state")
        elif isinstance(status, str):
            state = status

        if state is None or state in ("SUCCEEDED", "SUCCESS", "FAILED", "CANCELED", "CANCELLED"):
            return result

        time.sleep(base_delay * (2 ** attempt))

    return last_result

@st.cache_data(show_spinner=False)
def start_conversation_cached(question: str, space_id: str):
    # Cache by question + space to avoid repeated Genie calls
    return start_conversation(question)

@st.cache_data(show_spinner=False)
def get_query_result_cached(conversation_id, message_id, attachment_id):
    return get_query_result(conversation_id, message_id, attachment_id)

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Genie AI Analytics", layout="wide")

st.title("🧠 Databricks Genie AI")
st.caption("🔒 SQL Only • Table Output Only")

# Chat container
chat_container = st.container()

# Input at bottom (chat style)
question = st.chat_input("Ask a question (table answers only)...")

if question:
    with st.spinner("Genie generating SQL and executing query..."):
        try:
            normalized_question = question.strip().lower()
            cached_msg = st.session_state.question_cache.get(normalized_question)
            if cached_msg:
                msg = cached_msg
                cache_hit = True
            else:
                msg = start_conversation_cached(question, GENIE_SPACE_ID)
                st.session_state.question_cache[normalized_question] = msg
                cache_hit = False

            status = msg.get("status")

            if status == "FAILED":
                logging.error("Genie FAILED")
                st.error("Genie failed to process the request")
            else:
                attachments = msg.get("attachments") or []
                answer = msg.get("content", "")

                for att in attachments:
                    text = (att.get("text") or {}).get("content")
                    if text:
                        answer = text
                        break

                logging.info(f"Attachments: {attachments}")

                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "attachments": attachments,
                    "conversation_id": msg.get("conversation_id"),
                    "message_id": msg.get("message_id"),
                    "cache_hit": cache_hit,
                    "refreshed": False,
                    "viz_spec": extract_genie_viz_spec(attachments, answer)
                })

        except Exception as e:
            logging.error(str(e))
            st.error(f"Error: {str(e)}")

# -----------------------
# Display Chat + Tables
# -----------------------

with chat_container:
    for item in st.session_state.history:

        st.markdown("### 🧑 You")
        st.write(item["question"])

        st.markdown("### 🤖 Genie (SQL Engine)")
        st.write(item["answer"])

        attachments = item.get("attachments", [])

        table_rendered = False

        for att in attachments:
            if att.get("query"):
                attachment_id = att["attachment_id"]
                cache_key = f"{item['conversation_id']}:{item['message_id']}:{attachment_id}"
                cached = st.session_state.result_cache.get(cache_key)
                want_chart = is_chart_request(item.get("question", ""))
                want_table = DISPLAY_TABLE or is_table_request(item.get("question", ""))
                want_summary = is_summary_request(item.get("question", ""))
                genie_viz_spec = item.get("viz_spec")

                try:
                    if cached:
                        df = cached.get("df")
                        csv = cached.get("csv")
                        summary_text = cached.get("summary_text")
                        row_count = cached.get("row_count", 0)
                        col_count = cached.get("col_count", 0)
                        chart_payload = cached.get("chart_payload")
                    else:
                        result = get_query_result_with_retry(
                            item["conversation_id"],
                            item["message_id"],
                            attachment_id
                        )

                        statement = result.get("statement_response") or {}
                        manifest = statement.get("manifest") or {}
                        schema = (manifest.get("schema") or {}).get("columns", [])
                        rows = (statement.get("result") or {}).get("data_array", [])

                        row_count = len(rows) if rows else 0
                        col_count = len(schema) if schema else 0
                        summary_text = summarize_result(schema, rows)

                        df = None
                        csv = None
                        if want_table or ENABLE_CSV_DOWNLOAD or want_chart:
                            if schema and rows:
                                columns = [col["name"] for col in schema]
                                df = pd.DataFrame(rows, columns=columns)
                                if ENABLE_CSV_DOWNLOAD:
                                    csv = df.to_csv(index=False).encode("utf-8")

                        st.session_state.result_cache[cache_key] = {
                            "df": df,
                            "csv": csv,
                            "summary_text": summary_text,
                            "row_count": row_count,
                            "col_count": col_count,
                            "chart_payload": None,
                        }

                    if summary_text is not None:
                        if want_table and df is not None:
                            st.markdown("### 📊 Query Result")
                            st.dataframe(df, use_container_width=True)

                        if want_chart and df is not None and not df.empty:
                            st.markdown("### 📈 Chart")
                            if chart_payload is None:
                                chart_payload = build_chart_payload(
                                    df,
                                    item.get("question", ""),
                                    item.get("answer", ""),
                                    genie_viz_spec
                                )
                                if chart_payload is not None:
                                    st.session_state.result_cache[cache_key]["chart_payload"] = chart_payload

                            if chart_payload:
                                chart_df = pd.DataFrame(chart_payload.get("data") or [])
                                chart_type = chart_payload.get("chart_type")
                                title = chart_payload.get("title") or "Chart"
                                x = chart_payload.get("x")
                                y = chart_payload.get("y")
                                series = chart_payload.get("series") or []

                                if chart_type == "pie" and x in chart_df.columns and y in chart_df.columns:
                                    fig = px.pie(chart_df, names=x, values=y, title=title)
                                    st.plotly_chart(fig, use_container_width=True)
                                elif chart_type == "line" and series:
                                    fig = px.line(pd.DataFrame(chart_payload.get("data") or []), y=series, title=title)
                                    st.plotly_chart(fig, use_container_width=True)
                                elif x in chart_df.columns and y in chart_df.columns:
                                    if chart_type == "line":
                                        fig = px.line(chart_df, x=x, y=y, title=title)
                                    else:
                                        fig = px.bar(chart_df, x=x, y=y, title=title)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.write("No chartable columns returned.")
                            else:
                                st.write("No chartable columns returned.")

                        if want_summary and not want_table:
                            if SHOW_SUMMARY_HEADER:
                                st.markdown("### ✅ Result Summary")
                            st.write(summary_text)

                        # CSV Export
                        if ENABLE_CSV_DOWNLOAD and csv is not None:
                            st.download_button(
                                "⬇ Download CSV",
                                csv,
                                file_name="query_result.csv",
                                mime="text/csv",
                                key=f"download_{item['conversation_id']}_{item['message_id']}_{attachment_id}",
                            )

                        logging.info("Displayed SQL query result")
                        table_rendered = True

                except Exception as e:
                    logging.error(f"Query result fetch error: {str(e)}")
                    st.error("Failed to load query result")

        if not table_rendered:
            pass

        st.markdown("---")
