import json
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

_STORE: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]

def reset_session(session_id: str) -> None:
    """Used to demonstrate a fresh conversation with no carried-over state."""
    _STORE.pop(session_id, None)

def _invoke_crew(payload: dict, config=None) -> str:
    from agents import process_query_with_crew
    query = payload.get("input", "")
    history = payload.get("history", [])
    
    # Crew returns a dict. Convert to a JSON string so LangChain's 
    # RunnableWithMessageHistory can easily append it to the chat history.
    result_dict = process_query_with_crew(query, history)
    return json.dumps(result_dict)

_base_runnable = RunnableLambda(_invoke_crew)

conversational_agent = RunnableWithMessageHistory(
    _base_runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def ask_with_memory(session_id: str, query: str) -> dict:
    """Invokes the memory chain and parses the JSON string back to a dict for main.py."""
    raw_str = conversational_agent.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        return {"status": "error", "response": "Memory parse failed."}