# 🧠 SkillChain

SkillChain 是一个面向工程与研究的 **Agent Skill Framework**，目标不是做一个“能聊天就行”的 demo，而是：

- 把 **能力（Skills）** 做成显式、可复用的 Python 类  
- 把 **决策（Router）** 与 **执行（Skill）** 分离  
- 用 **Agent Runtime** 串起路由、执行和记忆  
- 用 **Workflows** 定义多步策略  
- 所有逻辑都可以 **阅读、调试、测试和扩展** 🛠️

> 🧩 心智模型：  
> Skills = functions（能力原语）  
> Router = policy（决策策略）  
> Agent = runtime（执行引擎）  
> Workflow = program（任务程序）

---

## 🏗️ 总体架构

整体分层如下（代码中也按照这个结构组织）：

> LLM Adapter  
> ⬇️  
> Skills（能力原语）  
> ⬇️  
> Skill Router（路由策略）  
> ⬇️  
> Agent Runtime（执行 + 显式 Memory）  
> ⬇️  
> Workflows（任务/策略）

对应的目录结构：

<details open>
<summary><strong>📂 点击查看完整目录结构 (Input Dataset)</strong></summary>
  
skillchain/
  __init__.py

  adapters/          # LLM 适配层
    __init__.py
    base.py
    openrouter.py

  skills/            # 能力原语（Skill）
    __init__.py
    base.py
    echo.py

  router/            # 决策层（选择哪个 Skill）
    __init__.py
    base.py
    llm_semantic.py
    rule_based.py

  memory/            # 显式记忆层
    __init__.py
    short_term.py
    long_term.py

  agent/             # Agent Runtime（执行器）
    __init__.py
    runtime.py

  workflows/         # 多步策略与编排
    __init__.py
    base.py
    single_step.py

  cli.py             # 最小可运行入口（命令行）

  types.py           # SkillCall / SkillResult 等核心类型

tests/
  test_runtime_keyword_router.py  # 关键执行链路测试

</details>

---

## ✨ 核心设计原则（在代码里是怎么体现的）

### 1. Agent ≠ Prompt 🧠

项目中没有 “一个超长 system prompt 控制整个 agent”的写法。

Agent 行为体现在：

- **skillchain/agent/runtime.py** 中的 `AgentRuntime`

- `skillchain/workflows/` 中的各种 `Workflow` 实现

Prompt 只出现在：

路由层（例如 skillchain/router/llm_semantic.py 用 LLM 做选择）

> （未来可选）某些具体的 Skill 内部

也就是说：**决策和策略是代码级别的，不是 prompt 拼接出来的黑箱。**

### 2. Skills 是一等公民 🧰
**所有 Skill 必须继承统一的基类**，skillchain/skills/base.py：
```python
# skillchain/skills/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SkillContext:
    """
    Execution context passed into a skill.

    Note: skills must not manage memory; this is read-only context from the agent runtime.
    """
    request_id: str
    short_term_context: Dict[str, Any]


class Skill(ABC):
    """
    Capability primitive.

    Rules:
      - Single responsibility
      - Explicit input/output
      - No routing
      - No memory management
      - No calling other skills directly
    """

    name: str
    description: str

    def __init__(self, *, name: Optional[str] = None, description: Optional[str] = None) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description

    @abstractmethod
    def run(self, *, skill_input: Dict[str, Any], ctx: SkillContext) -> Dict[str, Any]:
        raise NotImplementedError
```
约束（在代码注释中也写死了）：
- ✅ 单一职责：一个 Skill 做好一件事
- ✅ 显式输入输出：skill_input / 返回 Dict，可测试
- ❌ 不做路由
- ❌ 不直接管理 memory
- ❌ 不直接调用其他 Skill
  
**示例 Skill：EchoSkill（参考实现）**，skillchain/skills/echo.py：
```python
# skillchain/skills/echo.py
from typing import Any, Dict
from .base import Skill, SkillContext


class EchoSkill(Skill):
    """
    Reference skill: returns the provided input and a small ctx snapshot.
    """

    name = "echo"
    description = "Echoes input payload for debugging and pipeline verification."

    def run(self, *, skill_input: Dict[str, Any], ctx: SkillContext) -> Dict[str, Any]:
        return {
            "echo": skill_input,
            "ctx": {
                "request_id": ctx.request_id,
                "short_term_context_keys": sorted(list(ctx.short_term_context.keys())),
            },
        }
```
这相当于一个“示范技能”：
- 可以验证路由 → 执行 → memory 更新的整条链路
- 不承载业务逻辑，方便后续你换上自己的 Skills

### 3. Decision ≠ Execution 🧭
路由逻辑（选哪个 Skill）和执行逻辑（Skill.run）是完全分开的。

**路由抽象：SkillRouter**，skillchain/router/base.py：
```python
# skillchain/router/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from skillchain.types import SkillCall


@dataclass(frozen=True)
class RouterContext:
    """
    Context passed into a router to make a selection.

    Routers may use LLMs/prompts, but they must not execute tasks (no side effects beyond selection).
    """
    short_term_context: Dict[str, Any]
    long_term_summary: Optional[str] = None


class SkillRouter(ABC):
    """Selects which skill should run next, given a user goal and available skills."""

    @abstractmethod
    async def select_skill(
        self,
        *,
        user_intent: str,
        available_skills: List[Dict[str, str]],
        ctx: RouterContext,
    ) -> SkillCall:
        raise NotImplementedError
```
这里返回的是一个显式的数据结构 SkillCall（定义在 `skillchain/types.py`）：
```python
# skillchain/types.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

JSONDict = Dict[str, Any]


@dataclass(frozen=True)
class SkillCall:
    """A decision artifact: which skill to run and with what input payload."""
    skill_name: str
    skill_input: JSONDict
    rationale: Optional[str] = None
```
本地路由：KeywordRouter（无需 LLM），skillchain/router/rule_based.py：
```python
# skillchain/router/rule_based.py
from typing import Dict, List
from skillchain.types import SkillCall
from .base import RouterContext, SkillRouter


class KeywordRouter(SkillRouter):
    """
    Minimal local router (no LLM): picks a skill by keyword match.
    """

    def __init__(self, *, default_skill: str) -> None:
        self._default = default_skill

    async def select_skill(
        self,
        *,
        user_intent: str,
        available_skills: List[Dict[str, str]],
        ctx: RouterContext,
    ) -> SkillCall:
        _ = ctx  # selection-only; keep signature consistent
        names = {s["name"] for s in available_skills}

        intent_lower = user_intent.lower()
        for s in available_skills:
            name = s["name"]
            if name.lower() in intent_lower:
                return SkillCall(skill_name=name, skill_input={"text": user_intent}, rationale="keyword")

        if self._default not in names:
            # fallback to first available skill
            chosen = available_skills[0]["name"] if available_skills else self._default
            return SkillCall(skill_name=chosen, skill_input={"text": user_intent}, rationale="fallback-first")

        return SkillCall(skill_name=self._default, skill_input={"text": user_intent}, rationale="default")
```
特点：
- 完全不调用 LLM，适合本地快速开发、跑单测
- 只决定 skill 名字 + 输入参数，不执行任何 Skill
**LLM 语义路由：LLMSemanticRouter**，skillchain/router/llm_semantic.py：
```python
# skillchain/router/llm_semantic.py
import json
from typing import Any, Dict, List
from skillchain.adapters.base import ChatMessage, LLMAdapter
from skillchain.types import SkillCall
from .base import RouterContext, SkillRouter


class LLMSemanticRouter(SkillRouter):
    """
    Semantic router that uses an LLM to choose a skill.
    """

    def __init__(self, *, llm: LLMAdapter, model: str) -> None:
        self._llm = llm
        self._model = model

    async def select_skill(
        self,
        *,
        user_intent: str,
        available_skills: List[Dict[str, str]],
        ctx: RouterContext,
    ) -> SkillCall:
        system = (
            "You are a skill router. Select exactly one skill.\n"
            "Return STRICT JSON with keys: skill_name, skill_input, rationale.\n"
            "Rules: choose only from available skills; do not execute tasks; no prose outside JSON."
        )
        skills_json = json.dumps(available_skills, ensure_ascii=False)
        user = (
            f"User intent:\n{user_intent}\n\n"
            f"Available skills (JSON):\n{skills_json}\n\n"
            f"Short-term context keys:\n{sorted(list(ctx.short_term_context.keys()))}\n"
        )
        if ctx.long_term_summary:
            user += f"\nLong-term memory summary:\n{ctx.long_term_summary}\n"

        resp = await self._llm.chat(
            model=self._model,
            messages=[ChatMessage(role="system", content=system),
                      ChatMessage(role="user", content=user)],
            temperature=0.0,
            max_tokens=300,
        )
        data = json.loads(resp.content)
        return SkillCall(
            skill_name=str(data["skill_name"]),
            skill_input=dict(data.get("skill_input") or {}),
            rationale=data.get("rationale"),
        )
```
这里 Router 使用 LLM，但依然只输出一个 `SkillCall`，不执行任何任务。

### 4. Agent Runtime：执行 + 显式 Memory 🏃‍♂️
Agent 是一个很薄的 runtime，负责：
1. 找出有哪些 Skill 可用
2. 调用 Router 选 Skill
3. 执行 Skill
4. 用显式的 Memory 记录决策与结果

skillchain/agent/runtime.py：
```python
# skillchain/agent/runtime.py
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from skillchain.memory.long_term import LongTermMemory
from skillchain.memory.short_term import ShortTermMemory
from skillchain.router.base import RouterContext, SkillRouter
from skillchain.skills.base import Skill, SkillContext
from skillchain.types import SkillCall, SkillResult


@dataclass(frozen=True)
class AgentConfig:
    """Agent runtime config."""
    agent_name: str = "skillchain-agent"


class AgentRuntime:
    """
    Agent runtime: router -> execute selected skill -> update explicit memory.
    """

    def __init__(
        self,
        *,
        router: SkillRouter,
        skills: Iterable[Skill],
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self._router = router
        self._skills: Dict[str, Skill] = {s.name: s for s in skills}
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term
        self.config = config or AgentConfig()

    def list_skills(self) -> Dict[str, str]:
        return {name: getattr(skill, "description", "") for name, skill in self._skills.items()}

    async def step(self, *, user_intent: str) -> SkillResult:
        available = [{"name": n, "description": d} for n, d in self.list_skills().items()]
        router_ctx = RouterContext(
            short_term_context=self.short_term.context,
            long_term_summary=self.long_term.get_summary() if self.long_term else None,
        )
        call: SkillCall = await self._router.select_skill(
            user_intent=user_intent,
            available_skills=available,
            ctx=router_ctx,
        )
        self.short_term.record_decision(call)

        skill = self._skills.get(call.skill_name)
        if not skill:
            result = SkillResult(
                skill_name=call.skill_name,
                output={},
                success=False,
                error=f"Unknown skill selected by router: {call.skill_name}",
            )
            self.short_term.record_result(result)
            return result

        ctx = SkillContext(request_id=str(uuid.uuid4()), short_term_context=self.short_term.context)
        try:
            output = skill.run(skill_input=call.skill_input, ctx=ctx)
            result = SkillResult(skill_name=skill.name, output=output, success=True)
        except Exception as e:
            result = SkillResult(skill_name=skill.name, output={}, success=False, error=str(e))

        self.short_term.record_result(result)
        return result
```
注意：
- Agent 只负责“调度 + 记录”，不写死任何策略逻辑
- 执行结果也被封装成 `SkillResult`（在 skillchain/types.py 中）：
```python
# skillchain/types.py
@dataclass(frozen=True)
class SkillResult:
    """An execution artifact: structured output produced by a skill."""

    skill_name: str
    output: JSONDict
    success: bool = True
    error: Optional[str] = None
```
### 5. 显式 Memory：短期 & 长期 🧠
**短期记忆：ShortTermMemory（单次运行上下文）**，skillchain/memory/short_term.py：
```python
# skillchain/memory/short_term.py
from dataclasses import dataclass, field
from typing import Any, Dict, List
from skillchain.types import SkillCall, SkillResult


@dataclass
class ShortTermMemory:
    """
    Execution-context memory for a single run/session.

    Explicit structure (not prompt concatenation).
    """

    context: Dict[str, Any] = field(default_factory=dict)
    decisions: List[SkillCall] = field(default_factory=list)
    results: List[SkillResult] = field(default_factory=list)

    def record_decision(self, call: SkillCall) -> None:
        self.decisions.append(call)

    def record_result(self, result: SkillResult) -> None:
        self.results.append(result)
```
特点：
- 用 `context / decisions / results` 显式保存运行轨迹
- 以后你可以写自己的 summarizer，而不用从 prompt 里解析

**长期记忆：LongTermMemory（JSON 持久化）**，skillchain/memory/long_term.py：
```python
# skillchain/memory/long_term.py
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LongTermMemory:
    """
    Persistent knowledge store.

    Intentionally simple JSON file backend to keep the framework inspectable.
    """

    path: Path
    data: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        self.data = json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_summary(self) -> Optional[str]:
        """
        Optional: small human/LLM-readable summary for routers.
        Keep it short and explicit.
        """

        if not self.data:
            return None
        keys = sorted(list(self.data.keys()))
        return f"LongTermMemory keys: {keys}"
```
### 6. LLM Adapter：OpenRouter 统一封装 🌐
任何用 LLM 的地方，都应该通过 `LLMAdapter` 抽象调用。
**抽象接口**，skillchain/adapters/base.py：
```python
# skillchain/adapters/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    raw: Optional[Dict[str, Any]] = None


class LLMAdapter(ABC):
    """
    Unified LLM interface.
    """

    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        raise NotImplementedError
```
**OpenRouter 实现**，skillchain/adapters/openrouter.py：
```python
# skillchain/adapters/openrouter.py
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .base import ChatCompletion, ChatMessage, LLMAdapter


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "skillchain"
    http_referer: Optional[str] = None


class OpenRouterAdapter(LLMAdapter):
    """
    OpenRouter adapter via HTTP.
    """

    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        if config is None:
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise ValueError("Missing OPENROUTER_API_KEY for OpenRouterAdapter")
            config = OpenRouterConfig(
                api_key=api_key,
                http_referer=os.getenv("OPENROUTER_HTTP_REFERER") or None,
                app_name=os.getenv("OPENROUTER_APP_NAME") or "skillchain",
            )
        self._config = config

    async def chat(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        url = f"{self._config.base_url}/chat/completions"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "X-Title": self._config.app_name,
        }
        if self._config.http_referer:
            headers["HTTP-Referer"] = self._config.http_referer

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # OpenAI-compatible schema
        content = data["choices"][0]["message"]["content"]
        return ChatCompletion(content=content, raw=data)
```
> 上层代码（如 `LLMSemanticRouter`）只依赖 `LLMAdapter`，不会直接绑定某个 SDK。
### 7. Workflows：策略编排（而非“巨型 prompt”）📋
Workflow 是在 Agent 之上做多步流程，而不是用 prompt 控制一切。
**抽象定义**，skillchain/workflows/base.py：
```python
# skillchain/workflows/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from skillchain.agent.runtime import AgentRuntime


class Workflow(ABC):
    """
    High-level multi-step strategy.
    """

    @abstractmethod
    async def run(self, *, agent: AgentRuntime, user_intent: str) -> Dict[str, Any]:
        raise NotImplementedError
```
**参考实现：单步 workflow**，skillchain/workflows/single_step.py：
```python
# skillchain/workflows/single_step.py
from typing import Any, Dict
from skillchain.agent.runtime import AgentRuntime
from .base import Workflow


class SingleStepWorkflow(Workflow):
    """Reference workflow: run exactly one router->skill step."""

    async def run(self, *, agent: AgentRuntime, user_intent: str) -> Dict[str, Any]:
        result = await agent.step(user_intent=user_intent)
        return {
            "result": result,
            "short_term": {
                "context": agent.short_term.context,
                "decisions": agent.short_term.decisions,
                "results": agent.short_term.results,
            },
        }
```

---

## ▶️ 运行说明（Windows / Cursor 友好）
### 1. 安装依赖（开发模式）
你当前环境下 `python` 不在 PATH，但有 Windows Launcher `py`，建议使用：
```bash
py -m pip install -e .[dev]
```
这会安装：主包 `skillchain`，开发依赖：`pytest`、`ruff`
### 2. 不依赖 LLM 的本地运行（KeywordRouter）
```bash
py -m skillchain.cli "echo hello" --router keyword
```
对应入口代码在skillchain/cli.py：
```bash
# skillchain/cli.py
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
```
默认只注册了 `EchoSkill`，因此整条链路是：
`user_intent` → `KeywordRouter` 选择 `echo` → `EchoSkill.run` → `ShortTermMemory` 记录 → 输出 JSON。
### 3. 使用 OpenRouter 做语义路由（可选）
设置环境变量：
```bash
setx OPENROUTER_API_KEY "your_api_key_here"
```
运行：
```bash
py -m skillchain.cli "echo hello" --router llm --model openai/gpt-4o-mini
```
此时会走：
- `OpenRouterAdapter.chat` 调用 OpenRouter API
- `LLMSemanticRouter` 用 LLM 挑一个 Skill（当前只有 `echo`）
### 4. 运行测试
```python
py -m pytest -q
```
当前有一个核心测试，tests/test_runtime_keyword_router.py：
```bash
# tests/test_runtime_keyword_router.py
import asyncio
from skillchain.agent.runtime import AgentRuntime
from skillchain.router.rule_based import KeywordRouter
from skillchain.skills.echo import EchoSkill


def test_agent_runtime_runs_echo_via_keyword_router() -> None:
    async def run() -> dict:
        agent = AgentRuntime(router=KeywordRouter(default_skill="echo"), skills=[EchoSkill()])
        result = await agent.step(user_intent="please echo this")
        assert result.success is True
        assert result.skill_name == "echo"
        assert result.output["echo"]["text"] == "please echo this"
        return result.output

    asyncio.run(run())
```
验证整条 pipeline：
> Router → Skill → Result → Memory

---

## 🧩 扩展指南（如何在现有代码上加东西）
### 1. 新增一个 Skill
1. 新建文件，例如 `skillchain/skills/code_search.py`
2. 继承 `Skill`：
```bash
from typing import Any, Dict
from .base import Skill, SkillContext


class CodeSearchSkill(Skill):
    name = "code_search"
    description = "Search codebase for symbols or patterns."

    def run(self, *, skill_input: Dict[str, Any], ctx: SkillContext) -> Dict[str, Any]:
        query = skill_input["query"]
        # ...实现你的逻辑...
        results = []
        return {"query": query, "results": results}
```
3. 在你的 runtime 里注册它（例如自定义 CLI 或脚本）：
```bash
skills = [EchoSkill(), CodeSearchSkill()]
agent = AgentRuntime(router=my_router, skills=skills, long_term=my_ltm)
```
### 2. 新增一个 Router
1. 新建 `skillchain/router/xxx.py`
2. 继承 `SkillRouter`，实现 `select_skill` 返回 `SkillCall`
3. 在构建 AgentRuntime 时注入你的 Router
### 3. 新增一个 Workflow（多步策略）
1. 新建 `skillchain/workflows/my_workflow.py`
2. 继承 `Workflow`，在 `run` 里多次调用 `await agent.step(...)`，或者基于 `ShortTermMemory` 设计策略

---

## 🚫 明确不做的事情（Anti-patterns）
为了保持 SkillChain 的“框架味”，强制避免以下模式：
- ❌ 用一个巨大无比的 prompt 控制整个 Agent 行为
- ❌ 在路由中直接执行任务（Router 必须只返回 SkillCall）
- ❌ Skill 之间互相直接调用，形成隐式依赖网
- ❌ 把 memory 通过 string 拼接方式偷偷注入到 prompt，变成“隐式状态”
- ❌ 写一个 if/else 大怪兽在一个脚本里直接决定所有逻辑

推荐的做法是：
- ✅ 新增 Skill 文件；
- ✅ 新增 Router 文件（或者扩展已有的 LLM/规则路由）；
- ✅ 在 Workflow 中组合多步 Agent 行为；
- ✅ 在 Memory 层显式记录和回放运行轨迹。

---

## 🤝 贡献与支持 (Contribution)

**Next AI PlantUML** 是一个开源项目，我们需要您的帮助让它变得更好！

*   **给个 Star** ⭐：如果您觉得这个项目对您有帮助，请点击右上角的 Star，这是对我最大的鼓励！
*   **提交 Issue** 🐛：发现 Bug 或有新功能建议？欢迎提交 Issue。
*   **提交 PR** 🧑‍💻：欢迎贡献代码，无论是修复 Bug 还是增加新特性。

---

## 👤 作者 (Author)

**Haoze Zheng**

*   🎓 **School**: Xinjiang University (XJU)
*   📧 **Email**: zhenghaoze@stu.xju.edu.cn
*   🐱 **GitHub**: [mire403](https://github.com/mire403)

---
