from __future__ import annotations
import asyncio
import argparse
import logging
import os
from typing import Optional

from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

from skills.chemistry import ChemistrySkill
from skills.economics import EconomicsSkill

from orchestration.tools import make_chemistry_tools, make_economics_tools
from orchestration.graph import build_react_agent, extract_final_answer

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="gpt-4.1-mini")
    p.add_argument("--token", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--token-env", default="OPENAI_API_KEY")
    return p.parse_args()

def resolve_token(args: argparse.Namespace) -> str:
    if args.token:
        return args.token
    v = os.getenv(args.token_env)
    if v:
        return v
    raise RuntimeError(f"Missing token: provide --token or set {args.token_env}")

async def main() -> int:
    init_logging(logging.INFO)
    args = parse_args()

    model = args.model
    token = resolve_token(args)
    base_url: Optional[str] = args.base_url

    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        chem = await manager.launch(ChemistrySkill)
        econ = await manager.launch(EconomicsSkill)

        tools = []
        tools.extend(make_chemistry_tools(chem))
        tools.extend(make_economics_tools(econ))

        agent = await build_react_agent(model=model, api_key=token, base_url=base_url, tools=tools)

        question = "What is the cost of benzene? Use SMILES c1ccccc1 and return the numeric value."
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

        answer = extract_final_answer(result)
        print(answer)

    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
