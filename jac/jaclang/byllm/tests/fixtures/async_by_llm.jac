"""Async `by llm()` codegen: an async def must emit `await Jac.acall_llm(...)`.

The test binds `llm` before calling.
"""

glob llm: any = None;

async def get_summary(text: str) -> str by llm();

def get_summary_sync(text: str) -> str by llm();
