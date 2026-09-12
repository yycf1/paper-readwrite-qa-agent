# 文献阅读复现 Agent · 一期规划（定稿）

> 定稿日期：2026-09-10
> 产品定位：**阅读助手 + 复现地基**
> 一期完成「检索 → 下载 → 解析 → 实验知识提取 → RAG 问答」全链路 CLI 工具；二期在此之上叠加论文复现能力。

---

## 1. 项目定位与成功标准

- **产品形态**：个人科研工作流 CLI 工具（`pa` 命令）。
- **核心定位**：不是又一个 ChatPDF。普通「单篇文档问答」是商品化能力，本项目的差异化在两处：
  1. **结构化实验知识（experiment.json）**：把论文的方法与实验提炼为机器可用的结构化产物，市面工具没有，这是二期复现功能最关键的输入；
  2. **阅读助手体验（report.md）**：每篇论文自动产出精读报告（含可复现性备注），把 agent 从「问答工具」升级为「阅读助手」。
- **成功标准**：自己的文献阅读流程**每周真实在用**（检索新论文 → 入库 → 生成精读笔记 → 随时问答），而不是「能演示」。
- **二期目标（一句话）**：以一期产物为输入，实现论文复现（复现规划 → 环境搭建 → 代码生成执行 → 结果比对）。
- **精力配比原则**：实验知识提取与精读报告的打磨占大头；RAG 检索做到「够用 + 可回归验证」即可，不追求花哨。

## 2. 范围

### 2.1 一期做

- OpenAlex / Europe PMC / arXiv 三源检索与合并去重（来源选型依据见 §3，按本机网络实测调整）；
- 开放获取全文 PDF 自动下载；本地文件导入（PDF / DOCX，含中文论文路径）；
- PDF / DOCX → 分节 Markdown 解析；
- LLM 结构化实验知识提取（experiment.json，含复现资产链接与置信度标注）；
- 每篇论文自动生成精读报告（report.md）；
- 结构感知分块 + Chroma 向量索引 + 带引用的多轮问答（REPL）；
- library.db 文献库账本：全流程状态记录、断点续跑、增量跳过；
- LangGraph 一键流水线 + chat 意图路由；
- 轻量 RAG 评测集与 `pa eval` 回归命令。

### 2.2 一期明确不做（防过度设计）

| 不做项 | 原因 / 替代方案 |
|---|---|
| `.doc` 老格式 | python-docx 仅支持 `.docx`；遇到 `.doc` 提示用 Word 另存，写进 README |
| 知网/万方自动下载 | 无公开 API 且有合规风险；中文论文走 `pa add` 本地导入 |
| S2 引用图谱 / 相关论文推荐 | 价值大但非必需，列为**二期第一个特性**；一期 S2 只做搜索 + OA 链接 |
| LangGraph 重型 checkpoint | 流程基本线性，用 library.db 状态跳过实现续跑；SqliteSaver 留给二期 chat 会话记忆 |
| Web UI / 多用户 / 部署 | CLI 先行，界面二期后再议 |
| 复现功能本体 | 仅保证产物接口兼容（见 §12） |

## 3. 已确认的关键决策

| 决策项 | 结论 |
|---|---|
| 文献来源 | **OpenAlex（主力索引）+ Europe PMC（生物医学）+ arXiv（适配器保留）**。2026-09-11 本机实测：Semantic Scholar API 可达但匿名池 429 拥挤，PubMed eutils 与 arXiv 连接被重置（网络屏蔽）；OpenAlex 与 Europe PMC 直连可用且均免 Key。S2 降为可选 fallback（申请 Key 或低峰使用 + 缓存）；arXiv 走可选代理，不可达时降级跳过并标注 |
| LLM / Embedding 平台 | 主：硅基流动（BAAI/bge-m3 embedding 长期免费、中英多语、8K 上下文；LLM 聚合 DeepSeek/Qwen/GLM，按量便宜）；备：智谱（glm-4-flash 免费档 + embedding-3）。两家均 OpenAI 兼容，`base_url + api_key + model` 配置化切换，不锁定任何一家 |
| 免费额度风险处理 | M0 第一个任务即实测定稿配置；OpenAI 兼容抽象保证换平台只改配置 |
| 交互形态 | CLI（Typer + rich），问答为交互 REPL |
| 编排框架 | LangGraph：节点薄、模块厚。六个功能模块独立成包、可单独命令调用，LangGraph 只负责串流水线与意图路由 |
| 向量库 | Chroma（本地零部署，Windows 友好） |
| 全局唯一键 | 单篇论文以 `source_id`（如 `arxiv:2401.12345`）为全局键，目录、账本、向量库元数据统一用它关联 |

## 4. 总体架构

```
                         用户（CLI 命令 / 问答 REPL）
                                  │
┌─────────────────────────────────▼──────────────────────────────┐
│                     LangGraph 编排层                             │
│        一键流水线 run ── 意图路由 chat ── 各命令单步调用            │
└───┬──────────┬──────────┬──────────┬──────────┬────────────────┘
    ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│① 检索   │ │② 下载   │ │③ 解析    │ │④ 知识提取│ │⑤ RAG问答 │
├────────┤ ├────────┤ ├─────────┤ ├─────────┤ ├─────────┤
│arXiv   │ │PDF落盘  │ │PDF→分节MD│ │experiment│ │分块/索引  │
│S2      │ │本地导入  │ │DOCX→MD  │ │.json     │ │检索/引用  │
│PubMed  │ │去重/限速│ │+精读报告 │ │+复现资产  │ │多轮REPL  │
└────────┘ └────────┘ └─────────┘ └─────────┘ └─────────┘
    └──────────┴──────────┴─────┬────┴──────────┴──────────┘
                                ▼
              ┌─────────────────────────────────┐
              │   library.db（文献库账本，SQLite） │
              │   discovered → downloaded       │
              │   → parsed → analyzed → indexed │
              └─────────────────────────────────┘

━━━ 二期预留槽位（一期只保证接口兼容，不实现）━━━
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐
│复现规划器 │ │环境搭建器│ │代码生成/执行│ │结果比对  │
│读 JSON 出│ │clone 仓库│ │LLM 写代码 │ │指标对比  │
│复现方案   │ │+依赖安装 │ │+运行      │ │+报告    │
└─────────┘ └─────────┘ └──────────┘ └─────────┘
   ▲ 二期四模块全部以 experiment.json + 复现资产链接 为输入
```

设计约束（全项目只有这一条硬约束）：**二期的全部输入收敛到 `experiment.json` 和复现资产链接。** 一期做这两个产物时宁可字段宁多勿缺、宁可标注置信度，也不输出自由文本。

## 5. 数据产物与目录结构

```
E:\文献阅读复现agent\
├── PLAN.md  README.md  config.yaml  .env.example  .gitignore
├── data/                        # 全部运行时产物（进 .gitignore）
│   ├── papers/{paper_id}/       # paper.pdf 等原始文件
│   ├── parsed/{paper_id}/       # full_text.md（分节 Markdown）
│   ├── knowledge/{paper_id}/    # experiment.json + report.md
│   ├── library.db               # 文献库账本（SQLite）
│   └── db/                      # Chroma 向量库
├── src/paper_agent/
│   ├── cli.py  config.py  models.py
│   ├── sources/                 # arxiv.py  semanticscholar.py  pubmed.py
│   ├── parsing/                 # pdf.py  docx.py  splitter.py
│   ├── extraction/              # experiment_flow.py  report.py（含 prompt）
│   ├── rag/                     # embedder.py  vectorstore.py  retriever.py  qa.py
│   ├── graph/                   # state.py  nodes.py  pipeline.py
│   └── utils/                   # ratelimit.py  cache.py  logger.py
└── tests/                       # 仅覆盖核心纯逻辑：去重、分块、限速、状态机
```

## 6. experiment.json Schema（概要）

```jsonc
{
  "paper_id": "arxiv:2401.12345",
  "title": "...",
  "research_goal": "一句话研究目标",
  "method": { "name": "...", "architecture": "...", "key_components": ["..."] },
  "datasets": [ { "name": "...", "source": "...", "split": "..." } ],
  "hyperparameters": { "learning_rate": "...", "batch_size": "..." },
  "experiment_steps": [ { "step": 1, "description": "...", "details": "..." } ],
  "metrics": [ { "name": "...", "value": "...", "baseline": "..." } ],
  "main_results": "...",
  "conclusions": "...",
  "repro_assets": {                 // 复现资产：二期复现功能的关键入口
    "code_url": ["https://github.com/..."],
    "dataset_urls": ["..."],
    "project_page": ["..."]
  },
  "reproducibility_notes": "可复现性备注：代码是否开源、数据是否公开、细节充分度",
  "confidence": { "field_name": "high|medium|low" },   // 逐字段置信度
  "unverified_claims": ["LLM 推断但文中未明确说明的内容"],
  "token_usage": { "prompt": 0, "completion": 0 }
}
```

- 展示时低置信字段标黄提醒；`unverified_claims` 防 LLM 编造实验步骤。
- 产出缓存到文件，重跑/换 prompt 不会重复付费。

## 7. report.md 结构（精读报告）

每篇论文一次 LLM 调用生成，人读优先：

1. **一句话总结**
2. **研究问题与动机**
3. **方法脉络**（面向「读懂」而非「评审」）
4. **实验设计**（数据集 / 基线 / 指标）
5. **主要结论与局限**
6. **可复现性备注**（代码 / 数据 / 作者细节充分度，与 experiment.json 的 repro_assets 呼应）

## 8. CLI 命令设计

| 命令 | 功能 | 说明 |
|---|---|---|
| `pa search "主题" --source arxiv --max 10 --year 2022-2026` | 检索 | 三源合并去重，表格列出候选（标题/年份/引用数/状态） |
| `pa download <ids\|all>` | 下载 | OA PDF 落盘；失败标状态不中断 |
| `pa add <本地路径>` | 本地导入 | 注册本地 PDF/DOCX 入库（中文论文、已有文献的入口） |
| `pa parse <ids\|all>` | 解析 | 产出 full_text.md，已解析自动跳过 |
| `pa analyze <ids\|all>` | 知识提取 | 产出 experiment.json + report.md |
| `pa index <ids\|all>` | 索引 | 结构感知分块 → Chroma，增量索引 |
| `pa ask "问题"` / `pa ask` | 问答 | 单次提问 / 进入多轮 REPL，支持限定某篇论文；答案带出处，范围外明确说「论文未提及」 |
| `pa run "主题"` | 一键流水线 | 搜索→下载→解析→分析→索引全自动，**无人值守选文策略**见 §9 M4 |
| `pa status` | 库总览 | 各论文生命周期状态一览 |
| `pa eval` | RAG 回归 | 跑内置评测集，报告命中率 |

配置：`config.yaml` 按任务配模型（提取/报告用免费 flash 档，问答用更好的模型）+ 无人值守参数（top_n、年份过滤、下载预算）+ 平台 `base_url/api_key/model`。

## 9. 里程碑拆解（总工期约 3～3.5 周，业余时间估算）

### M0 · 项目初始化与环境验证（0.5～1 天）
1. 仓库骨架、uv 环境、config.yaml + .env 体系、.gitignore（data/ 与 .env）；
2. 申请硅基流动/智谱 Key，**实测定稿**：embedding（BGE-M3）可用性与价格、LLM 各档位；
3. LangGraph hello-world 图，验证 Windows 环境无坑。
- **验收**：一条命令输出「平台连通 + 图执行成功」。

### M1 · 检索下载 + 文献库账本（2.5～3 天）
1. 三源 client：OpenAlex（主力检索：关键词 + DOI/引用数/OA 链接，mailto 礼貌池 + 请求缓存）、Europe PMC（生物医学检索 + OA 全文 XML 下载）、arXiv（搜索 + PDF，限速 1 req/3s 写死，走可选代理；不可达时降级跳过）；
2. 统一 Paper 模型与 `source_id` 全局键；**模糊标题去重**（归一化大小写标点，因 DOI 覆盖不全）；
3. **library.db 账本**：五态生命周期，所有命令围绕账本决定「跳过还是执行」；
4. CLI：`search`、`download`、`status`；
5. `config.yaml` 增加全局 proxy 配置（供 arXiv 等被屏蔽源使用）；`pa doctor` 增加数据源连通性检查（网络屏蔽可自诊断）。
- **验收**：主题词返回多源合并去重的候选列表，下载 PDF 落盘，失败条目有状态标记；重复执行不重复下载。

### M2 · 解析 + 本地导入 + 知识提取（3.5～4 天）
1. `pa add` 本地导入（PDF/DOCX 注册入库，接同一后续流程）；
2. PDF → 分节 Markdown（PyMuPDF 基线，剥离参考文献，自动识别章节）；DOCX 走同一出口；解析结果缓存；
3. experiment.json 提取：固定 schema + 逐字段置信度 + unverified_claims + 复现资产链接（正则 + LLM 兜底）；
4. report.md 精读报告生成；
5. 在 3 篇样例论文（三源各一）上迭代 prompt 至验收标准。
- **验收**：样例论文产出干净分节文本；实验 JSON 字段完整、无编造（推断内容有标注）；能从中拿到 GitHub/数据集链接；报告人读流畅。CLI：`add`、`parse`、`analyze`。

### M3 · RAG 问答 + 评测集（3.5～4 天）
1. 结构感知分块：按章节切（500～800 token、带重叠），元数据带 paper_id / 章节名 / 页码；
2. Chroma 增量索引，`index` 命令；
3. 问答链：检索 top-k → 带引用回答（论文 + 章节/页），范围外拒绝回答；多轮 REPL；
4. **评测集**：2～3 篇论文手写 20～30 个问答对（含期望出处），做成 `pa eval`，防止分块/换模型/prompt 改动悄悄劣化。
- **验收**：提问 baseline/指标/步骤等答案准确带出处；跨论文提问能聚合；`pa eval` 有量化报告。

### M4 · 编排整合与打磨（2～2.5 天）
1. LangGraph 流水线：`run` 全链路；**无人值守选文策略**：按相关度/引用数取 top-N（可配置）+ 年份过滤 + 下载预算上限；
2. 断点续跑：基于 library.db 状态跳过已完成节点（不引入重型 checkpoint）；
3. `chat` 意图路由（搜索/解析/问答自动选择）；
4. README、日志、token 用量记录、端到端演示。
- **验收**：新用户按 README 配好 Key，`pa run "主题"` 一条命令到能问答，全程无手工干预。

### M5 · 检索智能化与工具契约（2～3 天，二期前置）

背景：体验测试后确立方向——检索是全流水线入口，入口质量决定下游垃圾进垃圾出；当前 `pa search` 是纯确定性管道（LLM 不参与），可靠但不智能（中文 query 搜英文库基本失效）。

1. **查询优化器**：LLM 一次调用完成 翻译（中→英学术表述）/ 展开（缩写→全称、口语→术语）/ 分解（复合问题拆 1~3 个子查询，复用现有去重合并）；**失败兜底：优化失败原样直搜，绝不阻塞**；
2. **统一工具契约**：所有工具（数据源 client、检索、后续 LLM 调用点）统一返回 `{status: ok|retryable_error|degraded|fatal, data, error, retry_after}`；重试与降级由框架按 status 处理；每个工具补 MCP 风格 inputSchema（为二期 MCP server 打地基）；
3. **结果合理性校验**：结果进入下游前做廉价结构校验（0 结果/必填字段缺失/年份越界）→ 触发改写重搜一轮或明确告知，不默默传垃圾；
4. **相关性预筛**：下载前 LLM 对候选列表按用户意图打相关分，踢掉明显不相关的（省下载与 token）；
5. **选文可解释**：`pa run` 选文附一句理由（如「3 篇高引综述 + 2 篇 2024 最新方法」）。
- **验收**：中文主题词能搜到相关英文论文；不相关候选被预筛剔除；任一源故障不影响其它源出结果（用新契约重构验证）；优化器失败时搜索照常工作。

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| PDF 解析质量（双栏/公式/表格）——最大不确定性 | PyMuPDF 跑基线，不够再引入 MinerU（学术 PDF 专用开源解析器，依赖较重）；解析结果缓存，换解析器无需重下论文 |
| PubMed 全文墙 | 摘要永远可用；全文仅 PMC OA 子集；其余降级为「仅摘要入知识库」，状态透明告知 |
| 本机网络屏蔽（实测 arXiv / PubMed 不通、S2 匿名池拥挤） | 主力改用 OpenAlex + Europe PMC（免 Key 直连可用）；arXiv 走可选代理，不可达时降级；doctor 内置数据源连通检查 |
| embedding 免费额度变动 | OpenAI 兼容抽象层，换平台只改配置 |
| RAG 质量无感知劣化 | M3 评测集 + `pa eval` 回归 |
| LLM 编造实验步骤 | 逐字段置信度 + unverified_claims 标注机制，展示时提示 |

## 11. 次级默认决策（有异议随时改）

- 解析产物用 Markdown 而非纯文本（保留结构，利于分块与人工阅读）；
- 测试只覆盖核心纯逻辑（去重、分块、限速、状态机），不追求覆盖率；
- 命令名 `pa`（paper agent）；如嫌拗口可全局替换；
- 数据目录固定在项目内 `data/`；
- 中文论文一律走 `pa add` 导入。

## 12. 开发工作流约定（M0 起生效）

1. **里程碑级保存**：每完成一个里程碑 → 更新 README 进度表 + 撰写 `docs/devlog/mN.md` 开发笔记（决策与踩坑，短小）→ 一次 `feat(mN)` 提交 → 打注解标签 `milestone/mN`。
2. **子功能级提交**：里程碑内部每个可运行的小步（如 M1 的 arXiv client、library.db 账本）随手小提交，保持细粒度可回滚，不等里程碑攒大提交。
3. **提交信息格式**：`<type>(<mN>): 中文摘要`，type 取 feat / fix / docs / test / chore；正文可写关键决策与踩坑。
4. **文档两级**：README 面向使用（快速开始 + 进度表）；`docs/devlog/` 面向复盘（设计决策、踩坑记录）。
5. 可选：仓库推送 GitHub 做异地备份（涉及外发，由用户自行决定）。

## 13. 二期路线图（预留，一期只保证接口兼容）

1. **复现规划器**：读 experiment.json + repro_assets，生成复现方案（要跑什么实验、需要什么环境）；
2. **环境搭建器**：clone 代码仓库、识别依赖、自动装环境；
3. **代码生成/执行器**：按方案生成/修改代码并运行，记录日志；
4. **结果比对**：复现指标与论文 metrics 字段自动比对，产出复现报告。
5. 顺位插入：**MCP server**（把 search/ask/analyze/status 按二期契约暴露为 MCP 工具，接入 Claude Desktop 等 MCP 宿主——M5 已备好 inputSchema 地基）；S2 引用图谱与相关论文推荐（「找出这篇的 related work」）；chat 模式会话记忆（SqliteSaver）；Web UI。

---

*本文档为一期开工基线。开工顺序：M0 → M4，每个里程碑独立可验收。*
