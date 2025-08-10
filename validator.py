import argparse
import asyncio
from pathlib import Path

from loguru import logger

from apex.common.async_chain import AsyncChain
from apex.common.config import Config
from apex.services.deep_research.deep_research_langchain import DeepResearchLangchain
from apex.services.llm.llm import LLM
from apex.services.websearch.websearch_tavily import WebSearchTavily
from apex.validator.auto_update import autoupdate_loop
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
        "--no-autoupdate",
        action="store_true",
        default=False,
        help="Disable automatic updates (checks every hour) (default: enabled)",
    )
    args = parser.parse_args()
    return args


async def main() -> None:
    args = await read_args()
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

    # Start autoupdate task if enabled
    autoupdate_task = None
    if not args.no_autoupdate:
        logger.info("Autoupdate enabled - will check for updates every hour")
        autoupdate_task = asyncio.create_task(autoupdate_loop())

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
        # Cancel autoupdate task if it was started
        if autoupdate_task is not None:
            autoupdate_task.cancel()
            try:
                await autoupdate_task
            except asyncio.CancelledError:
                pass

        await chain.shutdown()
        await logger_db.shutdown()
        await miner_scorer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
