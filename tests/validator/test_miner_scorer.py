import asyncio
import json
import tempfile
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, call, patch

import aiosqlite
import pytest
import pytest_asyncio

from apex.common.async_chain import AsyncChain
from apex.validator.miner_scorer import SCORE_MA_WINDOW_HOURS, MinerScorer


class TestMinerScorer:
    """Test suite for MinerScorer class."""

    @pytest_asyncio.fixture
    async def temp_db(self) -> AsyncGenerator[Any, Any]:
        """Create a temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path: str = temp_file.name
        temp_file.close()

        try:
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute("""
                    CREATE TABLE discriminator_results (
                        query                 TEXT,
                        generator_hotkey      TEXT,
                        generator_result      TEXT,
                        generator_score       REAL,
                        discriminator_hotkeys TEXT,
                        discriminator_results TEXT,
                        discriminator_scores  TEXT,
                        timestamp            INTEGER,
                        processed            INTEGER DEFAULT 0,
                        PRIMARY KEY (query, generator_hotkey)
                    )
                """)
                await conn.commit()

            with patch("apex.validator.miner_scorer.MinerScorer._db") as mock_db:
                from contextlib import asynccontextmanager

                @asynccontextmanager
                async def mock_db_context():
                    async with aiosqlite.connect(db_path) as conn:
                        await conn.execute("PRAGMA foreign_keys = ON")
                        yield conn

                mock_db.return_value = mock_db_context()
                yield db_path
        finally:
            try:
                Path(db_path).unlink()
            except FileNotFoundError:
                pass

    @pytest.fixture
    def mock_chain(self) -> AsyncMock:
        """Create a mock AsyncChain for testing."""
        mock = AsyncMock(spec=AsyncChain)
        mock.set_weights = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def miner_scorer(self, mock_chain: AsyncMock) -> MinerScorer:
        """Create a MinerScorer instance with mocked chain."""
        return MinerScorer(chain=mock_chain, interval=1.0)

    async def insert_test_data(self, db_path: str, data: list[dict[str, Any]]) -> None:
        """Helper method to insert test data into the database."""
        # Safety check to ensure we have a proper file path string.
        if not isinstance(db_path, str) or "async_generator" in db_path:
            raise ValueError(f"Invalid db_path: {db_path!r}")

        async with aiosqlite.connect(db_path) as conn:
            for row in data:
                await conn.execute(
                    """
                    INSERT INTO discriminator_results
                    (query, generator_hotkey, generator_result, generator_score,
                     discriminator_hotkeys, discriminator_results, discriminator_scores,
                     timestamp, processed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("query", "test_query"),
                        row["generator_hotkey"],
                        row.get("generator_result", "test_result"),
                        row["generator_score"],
                        json.dumps(row["discriminator_hotkeys"]),
                        json.dumps(row.get("discriminator_results", [0.5] * len(row["discriminator_hotkeys"]))),
                        json.dumps(row["discriminator_scores"]),
                        row["timestamp"],
                        row.get("processed", 0),
                    ),
                )
            await conn.commit()

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test MinerScorer initialization."""
        mock_chain: AsyncMock = AsyncMock(spec=AsyncChain)
        scorer: MinerScorer = MinerScorer(chain=mock_chain, interval=300.0)

        assert scorer.chain is mock_chain
        assert scorer.interval == 300.0
        assert scorer._running is True

    @pytest.mark.asyncio
    async def test_init_default_interval(self) -> None:
        """Test MinerScorer initialization with default interval."""
        mock_chain: AsyncMock = AsyncMock(spec=AsyncChain)
        scorer: MinerScorer = MinerScorer(chain=mock_chain)

        assert scorer.interval == 22 * 60

    @pytest.mark.asyncio
    async def test_shutdown(self, miner_scorer: MinerScorer) -> None:
        """Test shutdown functionality."""
        assert miner_scorer._running is True
        await miner_scorer.shutdown()
        assert miner_scorer._running is False

    @pytest.mark.asyncio
    async def test_set_scores_empty_database(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test set_scores with empty database."""
        result: bool = await miner_scorer.set_scores()
        assert result is True
        miner_scorer.chain.set_weights.assert_called_once_with({})  # type: ignore

    @pytest.mark.asyncio
    async def test_set_scores_with_valid_data(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test set_scores with valid data in the time window."""
        current_time: int = int(time.time())
        test_data: list[dict[str, Any]] = [
            {
                "generator_hotkey": "gen_hotkey_1",
                "generator_score": 0.8,
                "discriminator_hotkeys": ["disc_hotkey_1", "disc_hotkey_2"],
                "discriminator_scores": [0.6, 0.7],
                "timestamp": current_time - 3600,
            },
            {
                "generator_hotkey": "gen_hotkey_2",
                "generator_score": 0.9,
                "discriminator_hotkeys": ["disc_hotkey_1", "disc_hotkey_3"],
                "discriminator_scores": [0.5, 0.8],
                "timestamp": current_time - 7200,
            },
        ]

        await self.insert_test_data(temp_db, test_data)
        result: bool = await miner_scorer.set_scores()
        assert result is True
        expected_rewards: dict[str, float] = {
            "gen_hotkey_1": 0.8,
            "disc_hotkey_1": 0.6 + 0.5,
            "disc_hotkey_2": 0.7,
            "gen_hotkey_2": 0.9,
            "disc_hotkey_3": 0.8,
        }
        miner_scorer.chain.set_weights.assert_called_once_with(expected_rewards)  # type: ignore

    @pytest.mark.asyncio
    async def test_set_scores_filters_old_data(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test that set_scores filters out data older than the time window."""
        current_time: int = int(time.time())
        cutoff_time: float = current_time - SCORE_MA_WINDOW_HOURS * 3600
        test_data: list[dict[str, Any]] = [
            {
                "generator_hotkey": "gen_hotkey_recent",
                "generator_score": 0.8,
                "discriminator_hotkeys": ["disc_hotkey_1"],
                "discriminator_scores": [0.6],
                "timestamp": current_time - 3600,
            },
            {
                "generator_hotkey": "gen_hotkey_old",
                "generator_score": 0.9,
                "discriminator_hotkeys": ["disc_hotkey_2"],
                "discriminator_scores": [0.7],
                "timestamp": cutoff_time - 3600,
            },
        ]

        await self.insert_test_data(temp_db, test_data)
        result: bool = await miner_scorer.set_scores()

        assert result is True

        expected_rewards: dict[str, float] = {"gen_hotkey_recent": 0.8, "disc_hotkey_1": 0.6}

        miner_scorer.chain.set_weights.assert_called_once_with(expected_rewards)  # type: ignore

    @pytest.mark.asyncio
    async def test_set_scores_deletes_old_records(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test that set_scores deletes old records from the database."""
        current_time: int = int(time.time())
        cutoff_time: float = current_time - SCORE_MA_WINDOW_HOURS * 3600

        test_data: list[dict[str, Any]] = [
            {
                "generator_hotkey": "gen_hotkey_recent",
                "generator_score": 0.8,
                "discriminator_hotkeys": ["disc_hotkey_1"],
                "discriminator_scores": [0.6],
                "timestamp": current_time - 3600,
            },
            {
                "generator_hotkey": "gen_hotkey_old",
                "generator_score": 0.9,
                "discriminator_hotkeys": ["disc_hotkey_2"],
                "discriminator_scores": [0.7],
                "timestamp": cutoff_time - 3600,
            },
        ]

        await self.insert_test_data(temp_db, test_data)

        async with aiosqlite.connect(temp_db) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM discriminator_results")
            count_before_row = await cursor.fetchone()
            assert count_before_row is not None
            count_before: int = count_before_row[0]
            assert count_before == 2

        await miner_scorer.set_scores()

        async with aiosqlite.connect(temp_db) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM discriminator_results")
            count_after_row = await cursor.fetchone()
            assert count_after_row is not None
            count_after: int = count_after_row[0]
            assert count_after == 1

            cursor = await conn.execute("SELECT generator_hotkey FROM discriminator_results")
            remaining_hotkey_row = await cursor.fetchone()
            assert remaining_hotkey_row is not None
            remaining_hotkey: str = remaining_hotkey_row[0]
            assert remaining_hotkey == "gen_hotkey_recent"

    @pytest.mark.asyncio
    async def test_set_scores_malformed_json(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test set_scores handles malformed JSON gracefully."""
        current_time: int = int(time.time())

        async with aiosqlite.connect(temp_db) as conn:
            await conn.execute(
                """
                INSERT INTO discriminator_results
                (query, generator_hotkey, generator_result, generator_score,
                 discriminator_hotkeys, discriminator_results, discriminator_scores,
                 timestamp, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "test_query",
                    "gen_hotkey_1",
                    "test_result",
                    0.8,
                    "invalid_json",
                    "[0.5]",
                    "[0.6, 0.7]",
                    current_time - 3600,
                    0,
                ),
            )
            await conn.commit()

        with pytest.raises(json.JSONDecodeError):
            await miner_scorer.set_scores()

    @pytest.mark.asyncio
    async def test_set_scores_chain_failure(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test set_scores when chain.set_weights fails."""
        current_time: int = int(time.time())
        test_data: list[dict[str, Any]] = [
            {
                "generator_hotkey": "gen_hotkey_1",
                "generator_score": 0.8,
                "discriminator_hotkeys": ["disc_hotkey_1"],
                "discriminator_scores": [0.6],
                "timestamp": current_time - 3600,
            }
        ]

        await self.insert_test_data(temp_db, test_data)

        miner_scorer.chain.set_weights.return_value = False  # type: ignore

        result: bool = await miner_scorer.set_scores()
        assert result is False

    @pytest.mark.asyncio
    async def test_set_scores_multiple_same_hotkey(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test set_scores correctly aggregates multiple scores for the same hotkey."""
        current_time: int = int(time.time())
        test_data: list[dict[str, Any]] = [
            {
                "generator_hotkey": "gen_hotkey_1",
                "generator_score": 0.8,
                "discriminator_hotkeys": ["shared_disc_hotkey"],
                "discriminator_scores": [0.6],
                "timestamp": current_time - 3600,
            },
            {
                "generator_hotkey": "gen_hotkey_2",
                "generator_score": 0.7,
                "discriminator_hotkeys": ["shared_disc_hotkey"],
                "discriminator_scores": [0.4],
                "timestamp": current_time - 7200,
            },
        ]

        await self.insert_test_data(temp_db, test_data)
        result: bool = await miner_scorer.set_scores()

        assert result is True

        expected_rewards: dict[str, float] = {
            "gen_hotkey_1": 0.8,
            "gen_hotkey_2": 0.7,
            "shared_disc_hotkey": 1.0,
        }

        miner_scorer.chain.set_weights.assert_called_once_with(expected_rewards)  # type: ignore

    @pytest.mark.asyncio
    async def test_start_loop_runs_scoring(self, miner_scorer: MinerScorer) -> None:
        """Test that start_loop runs the scoring process."""
        miner_scorer.set_scores = AsyncMock(return_value=True)  # type: ignore

        loop_task: asyncio.Task[None] = asyncio.create_task(miner_scorer.start_loop())

        await asyncio.sleep(1.5)
        # Should be enough for one iteration with interval=1.0.

        await miner_scorer.shutdown()
        await loop_task

        assert miner_scorer.set_scores.call_count >= 1

    @pytest.mark.asyncio
    async def test_start_loop_handles_set_scores_failure(self, miner_scorer: MinerScorer) -> None:
        """Test that start_loop handles set_scores failures gracefully."""
        miner_scorer.set_scores = AsyncMock(return_value=False)  # type: ignore

        # Mock the logger to capture log calls.
        with patch("apex.validator.miner_scorer.logger") as mock_logger:
            # Start the loop in the background.
            loop_task: asyncio.Task[None] = asyncio.create_task(miner_scorer.start_loop())

            # Wait a bit for at least one iteration.
            await asyncio.sleep(1.5)

            # Stop the loop.
            await miner_scorer.shutdown()
            await loop_task

            assert mock_logger.error.called, "Expected logger.error to be called."
            error_calls = [call.args[0] for call in mock_logger.error.call_args_list]
            assert any("Failed to set weights" in str(msg) for msg in error_calls), (
                f"Expected log message not found. Found: {error_calls}"
            )

    @pytest.mark.asyncio
    async def test_start_loop_respects_shutdown(self, miner_scorer: MinerScorer) -> None:
        """Test that start_loop respects the shutdown signal."""
        miner_scorer.set_scores = AsyncMock(return_value=True)  # type: ignore

        loop_task: asyncio.Task[None] = asyncio.create_task(miner_scorer.start_loop())

        await asyncio.sleep(0.1)

        await miner_scorer.shutdown()

        try:
            await asyncio.wait_for(loop_task, timeout=3.0)
        except TimeoutError:
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            pytest.fail("Loop did not exit within timeout after shutdown.")

        assert miner_scorer.set_scores.call_count <= 1

    @pytest.mark.asyncio
    async def test_db_context_manager(self) -> None:
        """Test the database context manager."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path: str = tmp.name

        try:
            # Test the context manager with a real database.
            with patch("apex.validator.miner_scorer.aiosqlite.connect") as mock_connect:
                mock_conn: AsyncMock = AsyncMock()
                mock_connect.return_value.__aenter__.return_value = mock_conn

                async with MinerScorer._db() as conn:
                    assert conn is mock_conn

                mock_connect.assert_called_once_with("results.db")

                mock_conn.execute.assert_has_calls(
                    [
                        call("PRAGMA journal_mode=WAL"),
                        call("PRAGMA synchronous=NORMAL"),
                        call("PRAGMA busy_timeout=15000"),
                        call("PRAGMA foreign_keys=ON"),
                    ]
                )
                assert mock_conn.execute.call_count == 4
        finally:
            Path(db_path).unlink()

    @pytest.mark.asyncio
    async def test_score_window_hours_constant(self) -> None:
        """Test that the SCORE_WINDOW_HOURS constant is properly used."""
        # Score MA cannot be lower than 12 hours.
        assert SCORE_MA_WINDOW_HOURS >= 12
        assert isinstance(SCORE_MA_WINDOW_HOURS, int) | isinstance(SCORE_MA_WINDOW_HOURS, float)

    @pytest.mark.asyncio
    async def test_float_conversion_in_aggregation(self, temp_db: str, miner_scorer: MinerScorer) -> None:
        """Test that scores are properly converted to floats during aggregation."""
        current_time: int = int(time.time())
        test_data: list[dict[str, Any]] = [
            {
                "generator_hotkey": "gen_hotkey_1",
                "generator_score": "0.8",
                "discriminator_hotkeys": ["disc_hotkey_1"],
                "discriminator_scores": ["0.6"],
                "timestamp": current_time - 3600,
            }
        ]

        await self.insert_test_data(temp_db, test_data)
        result: bool = await miner_scorer.set_scores()

        assert result is True

        # Verify the conversion worked.
        expected_rewards: dict[str, float] = {"gen_hotkey_1": 0.8, "disc_hotkey_1": 0.6}

        miner_scorer.chain.set_weights.assert_called_once_with(expected_rewards)  # type: ignore
