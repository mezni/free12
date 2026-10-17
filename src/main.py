# src/main.py
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from src.workflows.graph import graph

load_dotenv()


def main():
    user_prompt = (
        "Research recent updates on LangGraph multi-agent workflows "
        "and provide a structured summary."
    )

    # Thread configuration for persistence memory
    config = {"configurable": {"thread_id": "default-cli-thread"}}

    print("=" * 60)
    print("🚀 [relay-agent] Starting Workflow Execution with Persistence...")
    print(f"📝 User Query: {user_prompt}")
    print("=" * 60 + "\n")

    initial_state = {
        "messages": [HumanMessage(content=user_prompt)],
        "next_agent": "Supervisor",
    }

    try:
        # Pass config with thread_id to graph.stream
        for event in graph.stream(initial_state, config=config):
            for node_name, output in event.items():
                print(f"⚡ [Node Executed]: {node_name}")
                if "next_agent" in output:
                    print(f"🎯 Route Decision -> Next: {output['next_agent']}\n")
                if "messages" in output:
                    for msg in output["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for call in msg.tool_calls:
                                print(
                                    f"  🛠️  [Tool Requested]: {call['name']} with args {call['args']}"
                                )
                        elif isinstance(msg, ToolMessage):
                            print(f"  📥 [Tool Output Received]: {msg.content[:150]}...")
                        elif hasattr(msg, "content") and msg.content:
                            print(f"  💬 [Message]: {msg.content[:200]}...\n")

        print("=" * 60)
        print("✅ Workflow Execution Complete & Persisted to checkpoints.sqlite!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")


if __name__ == "__main__":
    main()
