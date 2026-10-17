# evals/test_evals.py
import json
import pathlib

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from evals.metrics.correctness import evaluate_correctness
from evals.metrics.tool_usage import evaluate_tool_selection
from src.workflows.graph import graph

DATASET_PATH = pathlib.Path(__file__).parent / "datasets" / "golden_queries.json"


def load_dataset():
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


@pytest.mark.parametrize("test_case", load_dataset(), ids=lambda tc: tc["id"])
def test_agent_evals(test_case):
    # 1. Execute system workflow with persistent thread config per test case
    initial_state = {
        "messages": [HumanMessage(content=test_case["input"])],
        "next_agent": "Supervisor",
    }

    # Required checkpointer config
    config = {"configurable": {"thread_id": f"test-thread-{test_case['id']}"}}

    final_state = graph.invoke(initial_state, config=config)

    # 2. Extract messages and executed tools
    messages = final_state["messages"]
    final_response = messages[-1].content

    actual_tools = [
        msg.name for msg in messages if isinstance(msg, ToolMessage) and hasattr(msg, "name")
    ]

    # 3. Evaluate Metrics
    tool_score = evaluate_tool_selection(actual_tools, test_case["expected_tools"])
    correctness_score = evaluate_correctness(
        test_case["input"], test_case["expected_facts"], final_response
    )

    # 4. Assert Quality Thresholds
    assert tool_score == 1.0, f"Failed tool selection for {test_case['id']}"
    assert correctness_score >= 0.8, (
        f"Correctness score {correctness_score} below threshold for {test_case['id']}"
    )
