import pytest


@pytest.fixture(autouse=True)
def run_api_pipeline_inline_in_api_unit_tests(request, monkeypatch):
    """Keep API unit fixtures in-process; production uses a killable worker."""
    module_name = getattr(request.module, "__name__", "")
    if not module_name.endswith(("test_api", "test_dynamic_unseen_claims")):
        return

    from api import main

    async def run_inline(text, image):
        return main._run_pipeline(text, image)

    monkeypatch.setattr(main, "_run_pipeline_with_deadline", run_inline)
