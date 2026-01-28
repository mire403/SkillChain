from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from skillchain.agent.runtime import AgentRuntime
from skillchain.memory.long_term import LongTermMemory
from skillchain.router.llm_semantic import LLMSemanticRouter
from skillchain.router.rule_based import KeywordRouter
from skillchain.skills.echo import EchoSkill
from skillchain.workflows.single_step import SingleStepWorkflow


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skillchain", description="SkillChain minimal runner")
    p.add_argument("intent", nargs="?", default="echo: hello", help="user intent to route+execute")
    p.add_argument(
        "--router",
        choices=["keyword", "llm"],
        default="keyword",
        help="routing mode (keyword is local; llm uses OpenRouter)",
    )
    p.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model id")
    p.add_argument("--memory", default=".skillchain/memory.json", help="long-term memory JSON path")
    return p


async def _amain(args: argparse.Namespace) -> int:
    skills = [EchoSkill()]
    ltm = LongTermMemory(path=Path(args.memory))
    ltm.load()

    if args.router == "llm":
        from skillchain.adapters.openrouter import OpenRouterAdapter

        router = LLMSemanticRouter(llm=OpenRouterAdapter(), model=args.model)
    else:
        router = KeywordRouter(default_skill="echo")

    agent = AgentRuntime(router=router, skills=skills, long_term=ltm)
    wf = SingleStepWorkflow()
    out = await wf.run(agent=agent, user_intent=args.intent)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())

