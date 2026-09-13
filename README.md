# paper-agent · 文献阅读复现 Agent

把「论文检索 → 下载 → 解析 → 实验知识提取 → RAG 问答」做成一条本地 CLI 流水线；二期在此之上叠加论文复现能力。规划详见 [PLAN.md](PLAN.md)。

## 当前进度

| 里程碑 | 状态 |
|---|---|
| M0 初始化与环境验证 | ✅ 完成（硅基流动实测连通，BGE-M3 + DeepSeek-V3，`pa doctor` 全绿） |
| M1 检索下载 + 文献库账本 | ✅ 完成（OpenAlex/EuropePMC 实测通过，arXiv 待 proxy） |
| M2 解析 + 本地导入 + 知识提取 | ✅ 完成（PDF/DOCX 解析、experiment.json + report.md 端到端实测通过） |
| M3 RAG 问答 + 评测集 | ✅ 完成（索引 22 篇 142 块，`pa eval` 检索命中率 100%；答案生成实测待 .env 补 `ZHIPU_API_KEY`） |
| M4 编排整合 | ✅ 完成（`pa run` 全链路 + 断点续跑实测通过；analyze→问答两环待 `ZHIPU_API_KEY` 后端到端验收） |
| M5 检索智能化与工具契约 | ✅ 完成（中文查询优化、意图理解一体化、域外拒绝/方向引导、相关性预筛，实测全通过） |

## 快速开始

```powershell
# 1. 安装依赖（uv 管理）
python -m uv sync

# 2. 配置密钥：复制 .env.example 为 .env，填入 API Key
#    硅基流动 https://cloud.siliconflow.cn ｜ 智谱 https://open.bigmodel.cn

# 3. 环境自检（M0 验收命令：平台连通 + 图执行）
python -m uv run pa doctor
```

Windows 下若 `pa` 不在 PATH，用 `python -m uv run pa ...` 等价调用。

## 常用命令（M1 起）

```powershell
python -m uv run pa search "基于深度学习的医学影像分割" --max 10   # 检索入库（中文自动优化为英文检索词，--raw 跳过优化）
python -m uv run pa download --all          # 下载所有待处理论文的 OA PDF
python -m uv run pa add "路径\论文.pdf"      # 导入本地 PDF/DOCX（中文文献入口）
python -m uv run pa parse --all             # 解析为分节 Markdown
python -m uv run pa analyze --all           # LLM 提取 experiment.json + 精读报告
python -m uv run pa index --all             # 分块 + 向量化 + Chroma 索引（增量）
python -m uv run pa run "图神经网络 推荐"    # 一键流水线：检索→选文→下载→解析→提取→索引（LLM 相关性预筛）
python -m uv run pa ask "论文里用了哪些数据集？"   # 问答（带出处；无参进入多轮 REPL）
python -m uv run pa chat                    # 自然语言助手（检索/问答/状态/方向引导/域外拒绝）
python -m uv run pa eval                    # RAG 回归评测（--with-llm 加测答案质量）
python -m uv run pa status                  # 文献库状态总览
python -m uv run pa doctor                  # 环境自检（平台 + 数据源）
```

- 论文 id 可用片段（如 `W3217045679`、`PMC13451302`），多源结果自动去重；
- `pa chat` 支持带条件的自然语言检索（「找 2023 年以后的 XX 综述，前 20 篇」，自动抽取年份/数量并翻译成英文检索词）、方向推荐（「不知道看什么论文」）；与科研无关的请求会被礼貌拒绝；
- `pa run` 的无人值守选文带 LLM 相关性预筛与选文理由（显示在流水线 notes 中）；
- `pa run` 基于 library.db 状态断点续跑：中断后重跑同一主题自动跳过已完成阶段；
- `pa run` 无人值守参数在 `config.yaml` 的 `pipeline` 节（top_n / year_from / download_budget）；
- 无 OA 全文的论文标记「仅摘要」，其摘要也会以单块入知识库；
- `pa ask --paper <id片段>` 可限定单篇问答；范围外问题会明确回答「未提及」；
- `pa analyze`/`pa ask`/`pa run` 的知识提取需要 LLM Key；
- arXiv 在本机被网络屏蔽，检索自动降级；如需 arXiv，开启代理工具并在 `config.yaml` 的 `proxy` 填 `http://127.0.0.1:端口`。

## 配置说明

- `config.yaml`：平台 base_url / model、无人值守参数（top_n、年份过滤、下载预算）。切换硅基流动 ↔ 智谱只改此文件。
- `.env`：存放 API Key（变量名由各端点的 `api_key_env` 指定）。当前 `llm` 用智谱免费档（`ZHIPU_API_KEY`），`embedding` 用硅基流动 BGE-M3（`LLM_API_KEY`）。
