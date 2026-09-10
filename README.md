# paper-agent · 文献阅读复现 Agent

把「论文检索 → 下载 → 解析 → 实验知识提取 → RAG 问答」做成一条本地 CLI 流水线；二期在此之上叠加论文复现能力。规划详见 [PLAN.md](PLAN.md)。

## 当前进度

| 里程碑 | 状态 |
|---|---|
| M0 初始化与环境验证 | ✅ 完成（2026-09-11：硅基流动实测连通，BGE-M3 + DeepSeek-V3，`pa doctor` 全绿） |
| M1 检索下载 + 文献库账本 | ⬜ |
| M2 解析 + 本地导入 + 知识提取 | ⬜ |
| M3 RAG 问答 + 评测集 | ⬜ |
| M4 编排整合 | ⬜ |

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

## 配置说明

- `config.yaml`：平台 base_url / model、无人值守参数（top_n、年份过滤、下载预算）。切换硅基流动 ↔ 智谱只改此文件。
- `.env`：存放 API Key（变量名由 config.yaml 的 `api_key_env` 指定，默认 `LLM_API_KEY`）。
