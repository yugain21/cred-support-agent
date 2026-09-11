# memory.py
"""
Session memory via LangChain's InMemoryChatMessageHistory, keyed by
session_id. In-process only — doesn't need to survive a restart.
"""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def reset_session(session_id: str) -> None:
    """Clears a session's history, e.g. to start a fresh conversation."""
    _STORE.pop(session_id, None)


def _invoke_crew(payload: dict, config=None) -> str:
    # lazy import to avoid a circular import with agents.py
    from agents import process_query_with_crew
    query = payload["input"]
    result = process_query_with_crew(query)
    return str(result)


_base_runnable = RunnableLambda(_invoke_crew)

conversational_agent = RunnableWithMessageHistory(
    _base_runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def ask_with_memory(session_id: str, query: str) -> str:
    """Convenience wrapper main.py/eval.py call for a single turn."""
    return conversational_agent.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )


def transcript(session_id: str) -> list:
    history = get_session_history(session_id)
    return [{"role": m.type, "content": m.content} for m in history.messages]
