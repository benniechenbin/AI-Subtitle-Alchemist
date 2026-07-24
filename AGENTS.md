# AGENTS.md

## 1. 项目定位 / Project Role

本项目 `AI-Subtitle-Alchemist` 采用渐进式演进策略。可维护性、清晰边界、可测试性和当前需求优先于过度抽象、炫技或为未来不确定需求提前设计。

**Core Rule:** Do not over-engineer for enterprise, SaaS, distributed systems, multi-user scenarios, or large frameworks unless explicitly requested.

## 2. 当前阶段重点 / Current Focus

- **核心目标:** 构建AI驱动的自动化混剪脚本生成器
- **当前不优先 / Out of Scope:** 不引入与当前目标无关的大型重构、复杂框架或企业级能力。
- **Preference:** Prefer small, testable changes over large rewrites.

## 3. 架构边界与职责 / Architecture Boundaries

先识别项目现有目录、模块职责和依赖方向，再进行修改。不得为了迎合本模板而重构已有目录。

通用原则：

- 遵守项目已经明确的架构边界和依赖方向。
- 不得跨层直接访问其他模块的内部实现。
- 不得为了假设中的未来复用而提前拆分模块或引入复杂抽象。
- 配置读取、依赖组装和基础设施初始化应集中在明确的入口或组合层。
- 新增目录或架构层之前，必须说明现有结构为何无法承载当前需求。

本项目为 **独立应用 / Standalone Application**：

- 项目具有单一主要交付物，可以包含 CLI、UI、domain、service、storage 或 infrastructure 等内部模块。
- 不预设必须存在 `apps/*`、`packages/*` 或其他 Monorepo 目录。
- 优先沿用项目现有分层，不要为了形式统一而增加无实际职责的目录。
- `.env`、settings 和 runtime config 应由应用入口或组合层读取。
- 核心业务逻辑不应直接依赖 UI、CLI 或具体基础设施实现。
- 只有在已经出现明确复用需求时，才考虑提取独立 package。

## 4. 核心编码规范 / Coding & AI Rules

- **Plan First:** 修改代码前，先简要输出 implementation plan，说明将修改哪些文件、执行哪些步骤、主要风险是什么。Do not output hidden chain-of-thought.
- **Minimal Change:** 保持改动局部、清晰、可测试，不要扩大任务范围。
- **Reusability:** 优先复用现有 services, schemas, protocols, storage, tests。
- **Determinism:** Keep deterministic logic in code, not in LLM prompts.
- **Data Semantics:** 除非明确要求，否则不得改变既有数据口径、schema、view、hash、mapping 或输出格式。
- **When Unsure:** Choose the conservative path: small change, clear boundary, no new dependency, testable behavior.

**MUST:**

- Add or update tests when behavior changes.
- Preserve existing public APIs unless explicitly asked to change them.
- Explain architecture-level changes in the final response.
- State clearly if relevant tests were not run.

**SHOULD:**

- Prefer simple functions over premature abstractions.
- Prefer explicit rules over hidden magic.
- Prefer app-layer orchestration over package-layer coupling.

## 5. 绝对禁止事项 / THE DO NOTS

- **DO NOT** introduce cloud services, new databases, background services, or large frameworks for local/small needs.
- **DO NOT** silently change public APIs, schema, generated output formats, or data semantics.
- **DO NOT** let LLM prompts replace deterministic code for calculation, routing, parsing, or validation.
- **DO NOT** remove, weaken, or bypass tests to make changes pass.
- **DO NOT** hide failures behind broad `except Exception` without a clear fallback strategy.
- **DO NOT** make unrelated refactors in the same PR/change.

## 6. 测试与验证 / Testing

修改逻辑后，必须新增或更新相关测试。优先运行最小相关测试，再按需运行更大范围检查。

### Python Commands

```bash
uv run pytest [path] -q
uv run ruff check [path]
uv run ruff format [path] --check
uv run mypy [path]
```

如果无法运行测试，最终回复必须说明：

- 哪些测试没有运行。
- 为什么没有运行。
- 哪些风险需要人工确认。

## 7. 文档同步 / Documentation

当修改影响以下内容时，应同步更新相关文档：

- 架构边界变化。
- 数据流变化。
- CLI / UI 使用方式变化。
- 配置项变化。
- public API 变化。
- schema / storage / view / output format 变化。
- 版本路线变化。

优先检查：

- `README.md`
- `CHANGELOG.md`
- `ARCHITECTURE.md`
- `ROADMAP.md`
- 相关模块注释或测试说明

### 8.1 文档职责边界

- `README.md` 记录项目的介绍。
- `ARCHITECTURE.md` 记录项目架构和治理规则。
- `ROADMAP.md` 只用 `[x]` / `[ ]` 记录项目的宏观目标。
- `CHANGELOG.md` 记录项目的版本变化。
- `pyproject.toml` 是项目的唯一版本源；
- `docs/adr|rfc|prd` 记录项目详细的决策过程。

### 8.2 生命周期与检查

- ADR 保留长期决策历史；RFC/PRD 完成后将稳定事实同步到 README、ARCHITECTURE 或 CHANGELOG，并删除无维护价值的临时文档。
- 新增文档前必须先判断归属层级，并只维护对应事实来源，禁止为了说明而复制事实。
- 修改 public API、配置、schema、CLI、输出或架构边界时，只更新对应事实来源，不复制到多份文档。
- `src/**` 中的 prompt Markdown 是运行资源，不属于治理文档。
- 使用 `uv run python scripts/ci.py docs-check` 和 `uv run python scripts/ci.py version-check` 验证。

### 8.3 文档状态标注

- ADR / RFC / PRD 必须在文档开头元信息区标注 `状态：...`。状态值只允许使用以下枚举。

```text
PRD:
Draft → Review → Accepted → In Progress → Released
RFC:
Proposed → Accepted → Superseded / Rejected
ADR:
Proposed → Accepted → Deprecated / Superseded
```

### 8.4 文档更新范围

**Do not update documentation for unrelated changes.**

## 8. 项目专属规则 / Project-Specific Rules

在这里补充本项目逐步沉淀的专属规则：

## 9. AI 回复模板 / Required AI Response Format

完成代码修改后，最终回复必须包含以下结构：

> ### 变更总结 / Summary
>
> - **做了什么：** 说明修改了哪些文件和核心逻辑。
>
> - **为什么这么做：** 说明实现方式和关键权衡。
>
> - **测试与风险：** 列出已运行的测试命令；如果未运行，说明原因和风险。
>
> - **文档同步：** 说明是否更新了 README、CHANGELOG、ARCHITECTURE、ROADMAP 或其他相关文档。
>
> - **后续建议：** 如有必要，列出下一步最小可行改进。
>
