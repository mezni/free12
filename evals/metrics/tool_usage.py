# evals/metrics/tool_usage.py
def evaluate_tool_selection(actual_tools: list[str], expected_tools: list[str]) -> float:
    """Returns 1.0 if all expected tools were invoked, otherwise 0.0."""
    if not expected_tools:
        return 1.0

    called_set = set(actual_tools)
    expected_set = set(expected_tools)

    return 1.0 if expected_set.issubset(called_set) else 0.0
