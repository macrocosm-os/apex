"""Minimalistic miner for educational purposes only.

Walk through TODOs in this file to get the idea of what needs to be done.

================================================================================================================
                                                WARNING:
Do not run it on mainnet, this miner won't produce any yields and as a result the registration fee will be lost.
================================================================================================================
"""

import argparse
import random

import bittensor as bt
import requests
from aiohttp import web
from bittensor.core.extrinsics.serving import serve_extrinsic

NETUID_TESTNET = 61
NETUID_MAINNET = 1


async def handle_request(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        print(f"Incoming request:\n{body}")
    except Exception:
        body = {}
    if body.get("step") == "generator":
        # TODO: Replace with an actual completion.
        response = "This is a dummy generation from the base miner"
        print(f"Generator response: {response}")
        return web.Response(text=response)
    else:
        # TODO: Make actual classification for the given response. Validator class: 0; Miner class: 1.
        # body.get("query"): input query.
        # body.get("generation"): response generation.
        response = random.choice(["0", "1"])
        print(f"Discriminator response: {response}")
        return web.Response(text=response)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apex dummy miner")
    parser.add_argument(
        "--network",
        default="test",
        help="Network: test, finney.",
        type=str,
    )
    parser.add_argument(
        "--coldkey",
        help="Coldkey name.",
        type=str,
    )
    parser.add_argument(
        "--hotkey",
        help="Coldkey name.",
        type=str,
    )
    parser.add_argument(
        "--port",
        default=8080,
        help="Port.",
        type=int,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_request)
    args = read_args()
    wallet = bt.wallet(name=args.coldkey, hotkey=args.hotkey)
    subtensor = bt.subtensor(args.network)
    ip = requests.get("https://checkip.amazonaws.com").text.strip()

    netuid = NETUID_TESTNET if args.network == "test" else NETUID_MAINNET
    print(
        f"Serving miner on network: {args.network}, "
        f"netuid: {netuid}, "
        f"with wallet {args.coldkey}::{args.hotkey} "
        f"and address {ip}:{args.port}"
    )
    print("WARNING: THIS MINER IS DESIGNED ONLY FOR EDUCATIONAL PURPOSES, DO NOT RUN IT ON MAINNET.")
    serve_success = serve_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        ip=ip,
        port=args.port,
        protocol=4,
        netuid=netuid,
    )
    web.run_app(app, port=args.port)
