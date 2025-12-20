from __future__ import annotations
import asyncio
import argparse
import logging
import os
import json
from typing import Optional
from pathlib import Path

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

        smiles = "c1ccccc1"
        question = (
            "You are an assistant in a catalyst discovery stack. "
            f"Use the SMILES {smiles}. "
            "Call tools as needed to compute: (1) estimated cost, (2) color, (3) ionization energy. "
            "Then return a brief summary and a one-line decision: KEEP or REJECT."
        )
        
        logger.info("Query: %s", question)
        
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        
        from orchestration.graph import extract_tool_results, extract_tool_calls, extract_final_answer
        
        tool_calls = extract_tool_calls(result)
        tool_results = extract_tool_results(result)
        final_answer = extract_final_answer(result)
        
        logger.info("Tool calls: %s", tool_calls)
        logger.info("Tool results: %s", tool_results)
        logger.info("Final answer: %s", final_answer)
        
        # Simple derived decision for now
        decision = "KEEP" if "KEEP" in final_answer.upper() else ("REJECT" if "REJECT" in final_answer.upper() else "UNSPECIFIED")
        
        record = {
            "query": question,
            "smiles": smiles,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "final_answer": final_answer,
            "decision": decision,
        }

        Path("data").mkdir(exist_ok=True)
        with open("data/runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))
        
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
