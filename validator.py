import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from loguru import logger

from apex.common.async_chain import AsyncChain
from apex.common.config import Config
from apex.services.deep_research.deep_research_langchain import DeepResearchLangchain
from apex.services.llm.llm import LLM
from apex.services.websearch.websearch_tavily import WebSearchTavily
from apex.validator.logger_db import LoggerDB
from apex.validator.miner_sampler import MinerSampler
from apex.validator.miner_scorer import MinerScorer
from apex.validator.pipeline import Pipeline


async def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apex validator")
    parser.add_argument(
        "-c",
        "--config",
        # default="config/testnet.yaml",
        default="config/mainnet.yaml",
        help="Config file path (e.g. config/mainnet.yaml).",
        type=Path,
    )
    parser.add_argument(
        "--no-auto-update",
        action="store_true",
        help="Disable automatic git updates from main branch",
    )
    args = parser.parse_args()
    return args


async def update_from_git() -> bool:
    """
    Update the repository from the main branch.
    Returns True if successful or if no updates were needed, False if failed.
    """
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            logger.warning("Not in a git repository, skipping auto-update")
            return True

        logger.info("Checking for updates from main branch...")

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.stdout.strip():
            logger.warning("Uncommitted changes detected. Continuing with update, but you may want to commit or stash your changes.")

        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        current_branch = result.stdout.strip()
        
        # Only update if already on main branch
        if current_branch != "main":
            logger.info(f"Currently on branch '{current_branch}' (not main). Skipping auto-update to avoid disrupting your work.")
            logger.info("To enable auto-updates, switch to main branch or use --no-auto-update flag.")
            return True

        # Fetch latest changes
        result = subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            logger.error(f"Failed to fetch from origin: {result.stderr}")
            return False

        # Check if there are updates available
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        commits_behind = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        
        if commits_behind == 0:
            logger.info("Already up to date with main branch")
            return True

        logger.info(f"Found {commits_behind} new commit(s), pulling updates...")

        # Pull latest changes
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            logger.error(f"Failed to pull updates: {result.stderr}")
            return False

        logger.info("Successfully updated to latest version from main branch")
        logger.info("Note: You may want to restart the validator to ensure all changes are loaded")
        return True

    except Exception as e:
        logger.error(f"Error during git update: {e}")
        return False


async def main() -> None:
    args = await read_args()
    
    # Check for and pull updates from main branch (unless disabled)
    if not args.no_auto_update:
        update_success = await update_from_git()
        if not update_success:
            logger.warning("Git update failed, but continuing with validator startup")
    else:
        logger.info("Auto-update disabled via --no-auto-update flag")
    
    config = Config.from_file(path=args.config)

    chain = AsyncChain(**config.chain.kwargs)
    await chain.start()

    logger_db = LoggerDB(**config.logger_db.kwargs)
    asyncio.create_task(logger_db.start_loop())

    # logger_apex = LoggerApex(async_chain=chain)

    websearch = WebSearchTavily(**config.websearch.kwargs)

    miner_sampler = MinerSampler(chain=chain, logger_db=logger_db, **config.miner_sampler.kwargs)

    miner_scorer = MinerScorer(chain=chain, **config.miner_scorer.kwargs)
    asyncio.create_task(miner_scorer.start_loop())

    llm = LLM(**config.llm.kwargs)

    deep_research = DeepResearchLangchain(websearch=websearch, **config.deep_research.kwargs)

    pipeline = Pipeline(
        config=config,
        websearch=websearch,
        miner_sampler=miner_sampler,
        llm=llm,
        deep_research=deep_research,
        # logger_apex=logger_apex,
        **config.pipeline.kwargs,
    )
    try:
        await pipeline.start_loop()
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt caught, exiting validator")
    except BaseException as exc:
        logger.exception(f"Unknown exception caught, exiting validator: {exc}")
    finally:
        await chain.shutdown()
        await logger_db.shutdown()
        await miner_scorer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
