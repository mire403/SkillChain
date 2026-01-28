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
