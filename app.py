# app.py
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.workflows.graph import graph

load_dotenv()

st.set_page_config(
    page_title="relay-agent | Persistent Multi-Agent Workspace",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 relay-agent Workspace")
st.caption("Powered by LangGraph SQLite Persistence & GPT-4o.")

# Sidebar Thread Management for Long-Term Memory
st.sidebar.title("🧵 Session Memory")
thread_id = st.sidebar.text_input("Thread ID", value="default-session")
st.sidebar.info(f"Current active session: `{thread_id}`")

if st.sidebar.button("Clear Session History"):
    st.session_state.messages = []
    st.rerun()

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found. Please check your `.env` file.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like the agents to research or write?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        # Config mapping thread_id for SQLite checkpointer persistence
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "next_agent": "Supervisor",
        }

        final_response = ""
        try:
            with st.status(
                "⚡ Running persistent multi-agent workflow...", expanded=True
            ) as status:
                for event in graph.stream(initial_state, config=config):
                    for node_name, output in event.items():
                        st.write(f"🔄 **Node Executed:** `{node_name}`")
                        if "next_agent" in output:
                            st.write(f"🎯 **Routing Decision:** `{output['next_agent']}`")
                        if "messages" in output:
                            for msg in output["messages"]:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for call in msg.tool_calls:
                                        st.write(f"🛠️ *Tool Called:* `{call['name']}`")
                                elif isinstance(msg, ToolMessage):
                                    st.write("📥 *Fetched live web data*")
                                elif isinstance(msg, AIMessage) and msg.content:
                                    final_response = msg.content
                status.update(
                    label="✅ Completed & State Saved!",
                    state="complete",
                    expanded=False,
                )

            if not final_response:
                final_response = "Task completed successfully."

            response_placeholder.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
