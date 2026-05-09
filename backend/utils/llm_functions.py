import json
from typing import Callable, Dict, List


def call_llm_batch(
    strategy_fn: Callable[[str], str],
    todo_items: List[Dict],
    shared_examples: Dict,
    result_key: str,
    progress_cb=None,
) -> Dict[int, str]:
    """Send todo_items with few-shot examples to an LLM strategy and parse the response.

    Args:
        strategy_fn: Bound method to call (e.g. self.strategy.translate).
        todo_items: List of {'id': int, 'text': str} dicts to process.
        shared_examples: Class-level dict accumulating examples across files.
        result_key: Key to extract from each response item ('translation', 'gloss', …).
        progress_cb: Optional callback(current, total) for progress reporting.

    Returns:
        Dict mapping row index to result string.
    """
    if not todo_items:
        return {}

    examples = list(shared_examples.values())[:10]
    payload = json.dumps({"examples": examples, "items": todo_items}, ensure_ascii=False)
    response = json.loads(strategy_fn(payload))
    result = {item["id"]: item[result_key] for item in response["items"]}

    if progress_cb:
        progress_cb(len(todo_items), len(todo_items))

    return result
