import argparse
import asyncio
from pathlib import Path

from loguru import logger

from apex import __version__
from apex.common.async_chain import AsyncChain
from apex.common.config import Config
from apex.services.deep_research.deep_research_langchain import DeepResearchLangchain
from apex.services.llm.llm import LLM
from apex.services.websearch.websearch_tavily import WebSearchTavily
from apex.validator.logger_db import LoggerDB
from apex.validator.miner_sampler import MinerSampler
from apex.validator.miner_scorer import MinerScorer
from apex.validator.pipeline import Pipeline
from apex.validator.weight_syncer import WeightSyncer
from apex.validator.logger_wandb import LoggerWandb


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
    args = parser.parse_args()
    return args


async def main() -> None:
    args = await read_args()
    config = Config.from_file(path=args.config)
    logger.debug(f"Starting validator v{__version__} with config: {args.config}")

    chain = AsyncChain(**config.chain.kwargs)
    await chain.start()
    logger.debug(
        f"Connected to the chain netuid={chain.netuid} with coldkey '{chain.coldkey[:2]}***', "
        f"hotkey '{chain.hotkey[:2]}***'"
    )

    logger_db = LoggerDB(**config.logger_db.kwargs)
    asyncio.create_task(logger_db.start_loop())
    logger.debug(f"Started DB at: '{logger_db.db_path}'")

    logger_wandb = LoggerWandb(async_chain=chain)

    websearch = WebSearchTavily(**config.websearch.kwargs)
    logger.debug("Started web search tool")

    miner_sampler = MinerSampler(chain=chain, logger_db=logger_db, **config.miner_sampler.kwargs)
    logger.debug("Started miner sampler")

    weight_syncer = WeightSyncer(chain=chain, **config.weight_syncer.kwargs)
    await weight_syncer.start()
    logger.debug(
        f"Started weight synchronizer, receive enabled: {weight_syncer.receive_enabled}, "
        f"send enabled: {weight_syncer.send_enabled}, port: {weight_syncer.port}"
    )

    miner_scorer = MinerScorer(chain=chain, weight_syncer=weight_syncer, **config.miner_scorer.kwargs)
    asyncio.create_task(miner_scorer.start_loop())
    logger.debug(f"Started miner scorer with interval={miner_scorer.interval}")

    llm = LLM(**config.llm.kwargs)
    logger.debug("Started LLM provider")

    deep_research = DeepResearchLangchain(websearch=websearch, **config.deep_research.kwargs)
    logger.debug("Started Deep Researcher")

    pipeline = Pipeline(
        websearch=websearch,
        miner_sampler=miner_sampler,
        llm=llm,
        deep_research=deep_research,
        logger_wandb=logger_wandb,
        **config.pipeline.kwargs,
    )
    try:
        logger.debug("Starting pipeline loop...")
        await pipeline.start_loop()
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt caught, exiting validator")
    except BaseException as exc:
        logger.exception(f"Unknown exception caught, exiting validator: {exc}")
    finally:
        await chain.shutdown()
        await logger_db.shutdown()
        await miner_scorer.shutdown()
        await weight_syncer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
