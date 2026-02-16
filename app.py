import streamlit as st
import os
import logging
import datetime
import time
import re
from dotenv import load_dotenv
import pandas as pd
from databricks.sdk import WorkspaceClient

# LangGraph imports for memory management
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Optional, Dict, Any, List
import uuid

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

# -----------------------
# Memory Saver Configuration
# -----------------------
# Choose memory type: "sqlite" for persistent storage, "memory" for in-memory only
MEMORY_TYPE = os.getenv("MEMORY_TYPE", "sqlite")  # Options: "sqlite" or "memory"
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "genie_conversations.db")

# Keywords for table requests
TABLE_KEYWORDS = ("table", "tabular", "dataframe", "rows", "columns", "show data", "show table")

def is_table_request(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in TABLE_KEYWORDS)


ORDINAL_WORD_TO_NUMBER = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _ordinal_label(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _extract_requested_question_number(text: str) -> Optional[int]:
    lowered = text.lower()

    digit_match = re.search(r"\b(\d+)(st|nd|rd|th)?\b", lowered)
    if digit_match:
        try:
            return int(digit_match.group(1))
        except Exception:
            pass

    for word, number in ORDINAL_WORD_TO_NUMBER.items():
        if re.search(rf"\b{word}\b", lowered):
            return number

    return None


def maybe_answer_from_history(question: str, history: List[Dict[str, Any]]) -> Optional[str]:
    if not question:
        return None

    lowered = question.lower()
    if "question" not in lowered:
        return None

    asks_previous = bool(re.search(r"\b(previous|last)\b", lowered))
    requested_number = _extract_requested_question_number(lowered)
    asks_numbered_question = requested_number is not None and "question" in lowered
    generic_history_question = bool(re.search(r"\bwhat was my question\b", lowered))

    if not (asks_previous or asks_numbered_question or generic_history_question):
        return None

    questions = [item.get("question", "") for item in history if item.get("question")]
    if not questions:
        return "I do not have any earlier questions in this session yet."

    if asks_previous or generic_history_question:
        previous_question = questions[-1]
        return f'Your previous question was: "{previous_question}"'

    # Numbered question request
    target_index = requested_number - 1
    if target_index < 0 or target_index >= len(questions):
        return f"You have asked {len(questions)} question(s) in this session, so I cannot fetch question {_ordinal_label(requested_number)}."

    return f'Your {_ordinal_label(requested_number)} question was: "{questions[target_index]}"'

def _extract_sql_text(result: dict) -> str | None:
    if not isinstance(result, dict):
        return None

    statement = result.get("statement_response") or result.get("statement") or {}
    if isinstance(statement, dict):
        for key in ("statement", "sql", "query", "command"):
            val = statement.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        nested = statement.get("statement")
        if isinstance(nested, dict):
            for key in ("statement", "sql", "query", "command"):
                val = nested.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    for key in ("statement", "sql", "query", "command"):
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return None

# -----------------------
# LangGraph Memory Saver Functions
# -----------------------

@st.cache_resource(show_spinner=False)
def get_memory_saver():
    """
    Initialize and return the appropriate memory saver based on configuration.
    
    Returns:
        MemorySaver or SqliteSaver instance
    """
    try:
        if MEMORY_TYPE == "sqlite":
            # Use SqliteSaver for persistent storage
            logging.info(f"Initializing SqliteSaver with database: {SQLITE_DB_PATH}")
            # return SqliteSaver.from_conn_string(SQLITE_DB_PATH)
        else:
            # Use MemorySaver for in-memory storage (lost on restart)
            logging.info("Initializing MemorySaver (in-memory)")
            return MemorySaver()
    except Exception as e:
        logging.error(f"Failed to initialize memory saver: {e}")
        # Fallback to MemorySaver
        return MemorySaver()


def save_conversation_turn(
    checkpointer,
    thread_id: str,
    user_message: str,
    ai_response: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a conversation turn (user question + AI response) to LangGraph memory.
    
    Args:
        checkpointer: The memory saver instance
        thread_id: Unique thread identifier for this conversation
        user_message: User's question
        ai_response: AI's response
        metadata: Optional metadata (conversation_id, message_id, etc.)
    """
    try:
        # Prepare checkpoint data
        checkpoint_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ai_response}
            ],
            "metadata": metadata or {}
        }
        
        # Create config with thread_id
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "genie_conversation"
            }
        }
        
        # Save to checkpointer
        checkpointer.put(
            config=config,
            checkpoint=checkpoint_data,
            metadata={"step": len(get_conversation_history(checkpointer, thread_id)) + 1}
        )
        
        logging.info(f"Saved conversation turn to memory for thread: {thread_id}")
        
    except Exception as e:
        logging.error(f"Error saving conversation turn: {e}")


def get_conversation_history(
    checkpointer,
    thread_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history from LangGraph memory.
    
    Args:
        checkpointer: The memory saver instance
        thread_id: Thread identifier
        limit: Optional limit on number of messages to retrieve
        
    Returns:
        List of conversation turns
    """
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "genie_conversation"
            }
        }
        
        # Get all checkpoints for this thread
        history = []
        for checkpoint_tuple in checkpointer.list(config):
            checkpoint = checkpoint_tuple.checkpoint
            if isinstance(checkpoint, dict) and "messages" in checkpoint:
                history.append(checkpoint)
        
        # Sort by timestamp
        history.sort(key=lambda x: x.get("timestamp", ""))
        
        # Apply limit if specified
        if limit and len(history) > limit:
            history = history[-limit:]
        
        logging.info(f"Retrieved {len(history)} conversation turns for thread: {thread_id}")
        return history
        
    except Exception as e:
        logging.error(f"Error retrieving conversation history: {e}")
        return []


def get_all_thread_ids(checkpointer) -> List[str]:
    """
    Get all unique thread IDs from the memory store.
    
    Args:
        checkpointer: The memory saver instance
        
    Returns:
        List of thread IDs
    """
    try:
        thread_ids = set()
        
        # Iterate through all checkpoints
        for checkpoint_tuple in checkpointer.list({}):
            config = checkpoint_tuple.config
            if isinstance(config, dict):
                configurable = config.get("configurable", {})
                thread_id = configurable.get("thread_id")
                if thread_id:
                    thread_ids.add(thread_id)
        
        return sorted(list(thread_ids))
        
    except Exception as e:
        logging.error(f"Error retrieving thread IDs: {e}")
        return []


def clear_thread_history(checkpointer, thread_id: str):
    """
    Clear conversation history for a specific thread.
    
    Args:
        checkpointer: The memory saver instance
        thread_id: Thread identifier to clear
    """
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "genie_conversation"
            }
        }
        
        # Note: LangGraph checkpointers don't have a direct delete method
        # We'll need to implement this based on the specific checkpointer type
        logging.info(f"Cleared history for thread: {thread_id}")
        
    except Exception as e:
        logging.error(f"Error clearing thread history: {e}")


# -----------------------
# Logging Setup
# -----------------------
LOG_FILE = "genie_app.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("Genie Streamlit app started")

# Initialize memory saver
checkpointer = get_memory_saver()
logging.info(f"Memory saver initialized: {type(checkpointer).__name__}")

# -----------------------
# Session State
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "result_cache" not in st.session_state:
    st.session_state.result_cache = {}
if "question_cache" not in st.session_state:
    st.session_state.question_cache = {}
if "thread_id" not in st.session_state:
    # Generate a unique thread ID for this session
    st.session_state.thread_id = str(uuid.uuid4())
    logging.info(f"New session started with thread_id: {st.session_state.thread_id}")

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
    logging.debug("Raw query result received for attachment_id=%s", attachment_id)
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

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Genie AI Analytics", layout="wide")

st.title("Databricks Genie AI")
st.caption("SQL output with optional tables")

# Sidebar for memory management

# Chat container
chat_container = st.container()

# Input at bottom (chat style)
question = st.chat_input("Ask a question (table answers only)...")

if question:
    with st.spinner("Genie generating SQL and executing query..."):
        try:
            local_history_answer = maybe_answer_from_history(question, st.session_state.history)
            if local_history_answer:
                st.session_state.history.append({
                    "question": question,
                    "answer": local_history_answer,
                    "attachments": [],
                    "conversation_id": None,
                    "message_id": None,
                    "cache_hit": True,
                    "refreshed": False,
                })

                save_conversation_turn(
                    checkpointer=checkpointer,
                    thread_id=st.session_state.thread_id,
                    user_message=question,
                    ai_response=local_history_answer,
                    metadata={
                        "status": "local_history_answer",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                logging.info("Answered from local session history without calling Genie")
            else:
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
                    
                    # Save failed interaction to memory
                    save_conversation_turn(
                        checkpointer=checkpointer,
                        thread_id=st.session_state.thread_id,
                        user_message=question,
                        ai_response="[ERROR] Genie failed to process the request",
                        metadata={
                            "status": "failed",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                else:
                    attachments = msg.get("attachments") or []
                    answer = msg.get("content", "")

                    for att in attachments:
                        text = (att.get("text") or {}).get("content")
                        if text:
                            answer = text
                            break

                    logging.info(f"Attachments: {attachments}")

                    # Save to session state (existing behavior)
                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "attachments": attachments,
                        "conversation_id": msg.get("conversation_id"),
                        "message_id": msg.get("message_id"),
                        "cache_hit": cache_hit,
                        "refreshed": False,
                    })

                    # Save to LangGraph memory
                    save_conversation_turn(
                        checkpointer=checkpointer,
                        thread_id=st.session_state.thread_id,
                        user_message=question,
                        ai_response=answer,
                        metadata={
                            "conversation_id": msg.get("conversation_id"),
                            "message_id": msg.get("message_id"),
                            "cache_hit": cache_hit,
                            "attachment_count": len(attachments),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                    
                    logging.info(f"Conversation saved to LangGraph memory (thread: {st.session_state.thread_id})")

        except Exception as e:
            logging.error(str(e))
            st.error(f"Error: {str(e)}")
            
            # Save error to memory
            save_conversation_turn(
                checkpointer=checkpointer,
                thread_id=st.session_state.thread_id,
                user_message=question,
                ai_response=f"[ERROR] {str(e)}",
                metadata={
                    "status": "error",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )

# -----------------------
# Display Chat + Tables
# -----------------------

with chat_container:
    for item in st.session_state.history:

        st.markdown("### You")
        st.write(item["question"])

        st.markdown("### Genie (SQL Engine)")
        st.write(item["answer"])

        attachments = item.get("attachments", [])

        table_rendered = False

        for att in attachments:
            if att.get("query"):
                attachment_id = att["attachment_id"]
                cache_key = f"{item['conversation_id']}:{item['message_id']}:{attachment_id}"
                cached = st.session_state.result_cache.get(cache_key)
                want_table = DISPLAY_TABLE or is_table_request(item.get("question", ""))

                try:
                    if cached:
                        df = cached.get("df")
                        csv = cached.get("csv")
                        sql_text = cached.get("sql_text")
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
                        sql_text = _extract_sql_text(result)

                        df = None
                        csv = None
                        if want_table or ENABLE_CSV_DOWNLOAD:
                            if schema and rows:
                                columns = [col["name"] for col in schema]
                                df = pd.DataFrame(rows, columns=columns)
                                if ENABLE_CSV_DOWNLOAD:
                                    csv = df.to_csv(index=False).encode("utf-8")

                        st.session_state.result_cache[cache_key] = {
                            "df": df,
                            "csv": csv,
                            "sql_text": sql_text,
                        }

                    if sql_text or (want_table and df is not None) or (ENABLE_CSV_DOWNLOAD and csv is not None):
                        if sql_text:
                            st.markdown("### SQL Query")
                            st.code(sql_text, language="sql")
                        if want_table and df is not None:
                            st.markdown("### Query Result")
                            st.dataframe(df, use_container_width=True)


                        # CSV Export
                        if ENABLE_CSV_DOWNLOAD and csv is not None:
                            st.download_button(
                                "Download CSV",
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
