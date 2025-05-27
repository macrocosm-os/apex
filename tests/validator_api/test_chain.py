import pytest

from validator_api.chain.uid_tracker import TaskType, Uid, UidTracker


@pytest.fixture
def mock_shared_settings(monkeypatch: pytest.MonkeyPatch):
    class MockMetagraph:
        n: int = 3
        hotkeys: dict[int, str] = {0: "key0", 1: "key1", 2: "key2"}

    class MockSettings:
        METAGRAPH: MockMetagraph = MockMetagraph()

    monkeypatch.setattr("validator_api.chain.uid_tracker.shared_settings", MockSettings())


@pytest.fixture
def uid_tracker(mock_shared_settings: pytest.MonkeyPatch) -> UidTracker:
    tracker: UidTracker = UidTracker()
    tracker.resync()
    return tracker


@pytest.mark.asyncio
async def test_resync(uid_tracker: UidTracker) -> None:
    assert len(uid_tracker.uids) == 3
    assert uid_tracker.uids[0].hkey == "key0"
    assert uid_tracker.uids[1].hkey == "key1"
    assert uid_tracker.uids[2].hkey == "key2"


@pytest.mark.asyncio
async def test_set_query_attempt(uid_tracker: UidTracker):
    await uid_tracker.set_query_attempt(0, TaskType.CompletionInference)
    assert uid_tracker.uids[0].requests_per_task[TaskType.CompletionInference] == 1


@pytest.mark.asyncio
async def test_set_query_success(uid_tracker: UidTracker):
    await uid_tracker.set_query_success(0, TaskType.CompletionInference)
    assert uid_tracker.uids[0].success_per_task[TaskType.CompletionInference] == 1


@pytest.mark.asyncio
async def test_sample_reliable(uid_tracker: UidTracker):
    await uid_tracker.set_query_success(0, TaskType.CompletionInference)
    await uid_tracker.set_query_attempt(0, TaskType.CompletionInference)
    reliable: list[Uid] = await uid_tracker.sample_reliable(TaskType.CompletionInference, 1)
    assert len(reliable) == 1
    assert reliable[0].uid == 0


@pytest.mark.asyncio
async def test_sample_reliable_not_enough_uids(uid_tracker: UidTracker):
    # Set up one UID with a success rate above 0.8.
    await uid_tracker.set_query_success(0, TaskType.CompletionInference)
    await uid_tracker.set_query_attempt(0, TaskType.CompletionInference)

    # Set up another UID with a lower success rate 0.5.
    await uid_tracker.set_query_attempt(1, TaskType.CompletionInference)
    await uid_tracker.set_query_attempt(1, TaskType.CompletionInference)
    await uid_tracker.set_query_success(1, TaskType.CompletionInference)

    amount = 3

    # Request more UIDs than available with success_rate.
    reliable: list[Uid] = await uid_tracker.sample_reliable(
        TaskType.CompletionInference, amount=amount, success_rate=0.8
    )

    # Verify that the requested number of UIDs is returned.
    assert len(reliable) == amount
    # Verify that the top UIDs are returned, sorted by success rate.
    assert reliable[0].uid == 0
    assert await reliable[0].success_rate(TaskType.CompletionInference) == 1.0
    assert reliable[1].uid == 1
    assert await reliable[1].success_rate(TaskType.CompletionInference) == 0.5
    assert reliable[2].uid == 2
    assert await reliable[2].success_rate(TaskType.CompletionInference) == 0.0
