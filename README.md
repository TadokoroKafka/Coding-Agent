# 编程智能体

项目通过模型厂商的原生工具调用接口与大语言模型交互，智能体循环、本地工具、上下文管理、权限确认和错误处理均在本仓库中自行实现。

项目未使用 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等智能体框架，也未依赖托管的代码执行或文件服务。

## 主要功能

- 通过 OpenAI 兼容的 Chat Completions 接口使用 DeepSeek 原生工具调用。
- 自行实现五个本地工具：列出文件、读取文件、写入文件、精确替换和执行命令。
- 将所有文件与命令操作限制在指定工作区内，防止绝对路径、父目录和符号链接越界。
- 默认使用 `ask` 模式，对会产生副作用的操作进行明确确认。
- 提供适用于受控演示的受限 `auto` 模式。
- 仅重试模型 API 请求，并实现工具调用防重复和确定性循环终止。
- 上下文裁剪时完整保留工具调用与结果，并在裁剪后插入执行状态快照。
- 按字符安全截断 UTF-8 内容，跨平台处理命令超时，并记录经过脱敏的 JSONL 运行日志。
- 默认测试使用假模型客户端，不需要 API key，也不会访问网络。

## 系统架构

```text
用户任务
  -> 命令行参数与配置
  -> DeepSeek Chat Completions 请求
  -> 解析原生 tool_calls
  -> 权限与参数检查
  -> 本地工具单次执行
  -> 将结构化工具结果返回模型
  -> 进入下一模型步骤或输出最终回答
```

运行时由以下组件组成：

- `cli.py`：组装应用、读取任务、显示进度，并将运行结果映射为退出码。
- `model_client.py`：适配 DeepSeek 响应，仅对可恢复的 API 错误进行重试。
- `agent.py`：控制工具调用循环、防重复执行、错误计数与终止条件。
- `context.py`：只按完整交互组裁剪历史，并维护执行状态快照。
- `approval.py`：实现 `ask` 与受限 `auto` 权限策略。
- `tools/`：包含受工作区限制的文件工具和命令工具。
- `run_log.py`：将脱敏后的结构化日志写入 `<workspace>/.coding_agent/runs.jsonl`。

## 环境要求

- Python 3.12 或更高版本
- 真实模型请求需要 DeepSeek 兼容的 API key
- 支持 Windows、Linux 和 macOS

核心运行依赖只有 `openai`，测试额外使用 `pytest`。

## 安装

PowerShell：

```powershell
git clone https://github.com/TadokoroKafka/Coding-Agent.git
cd Coding-Agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Linux 或 macOS 使用以下命令激活虚拟环境：

```bash
source .venv/bin/activate
```

## 配置

凭据必须通过进程环境变量提供。不要将真实 key 写入源码、提交记录、文档、截图或演示视频。

PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "deepseek-v4-flash"
$env:DEEPSEEK_THINKING = "false"
```

支持的环境变量：

| 环境变量 | 是否必填 | 默认值 |
|---|---:|---|
| `DEEPSEEK_API_KEY` | 是 | 无 |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 否 | `deepseek-v4-flash` |
| `DEEPSEEK_THINKING` | 否 | `false` |

`.env.example` 是不含凭据的配置模板。程序读取进程环境变量，不会自动加载 `.env`。PyCharm 用户可以在运行配置的环境文件字段中选择未被 Git 跟踪的 `.env` 文件。

## 使用方法

为智能体创建或选择一个独立的项目目录，该目录必须已经存在。

```powershell
python -m coding_agent `
  --workspace D:\Project\agent-test-workspace `
  --approval-mode ask `
  --max-steps 20 `
  --verbose
```

程序随后提示输入任务：

```text
任务：检查项目，修复失败的测试，并总结修改内容
```

命令行参数：

| 参数 | 含义 |
|---|---|
| `--workspace PATH` | 必填。限定所有文件与命令操作的工作区目录。 |
| `--approval-mode ask` | 默认模式。读取自动执行，写入、替换和命令需要 `y/N` 确认。 |
| `--approval-mode auto` | 允许文件修改，命令仅允许 `python`、`pytest`、`git status` 和 `git diff`。 |
| `--max-steps N` | 模型请求次数上限，默认值为 `20`。 |
| `--verbose` | 显示经过脱敏的步骤、工具和结果状态。 |

`auto` 模式只用于受控演示，并不是完整的安全沙箱。请仅对允许被修改的工作区运行智能体。

## 本地工具

| 工具 | 行为 |
|---|---|
| `list_files` | 列出最多 500 个匹配的工作区文件，并忽略 Git、虚拟环境、缓存和运行日志。 |
| `read_file` | 按行范围读取 UTF-8 文本，并返回带行号的内容。 |
| `write_file` | 通过本地原子写入创建或完整覆盖 UTF-8 文本文件。 |
| `replace_in_file` | 仅当实际匹配次数等于 `expected_count` 时替换文本。 |
| `run_command` | 使用 `shell=False` 执行参数数组，并限制工作目录、输出长度和运行时间。 |

本地工具不会自动重试。这样可以防止超时或失败的命令被重复执行，避免产生重复副作用。

## 安全机制与终止条件

- 所有路径先经过解析，解析结果必须位于 `--workspace` 内。
- 拒绝绝对路径、`..` 和解析后越界的符号链接。
- 非 UTF-8 文件返回结构化错误，不擅自修改编码。
- Windows 命令超时后使用 `taskkill /T /F`，POSIX 系统则终止整个进程组。
- 重复的 `tool_call_id` 会被拒绝，同一规范化调用不能无限循环。
- 模型返回最终回答、达到步骤上限、重复调用、连续工具错误、连续非法调用、API 失败或用户中断时结束运行。
- 日志会脱敏敏感字段、认证信息、已知 API key 值和常见密钥模式。

## 测试

安装测试依赖并运行离线测试：

```powershell
python -m pip install -e ".[test]"
pytest
```

测试覆盖文件隔离、Unicode 处理、精确替换、权限策略、安全命令、超时清理、API 重试、模型输出解析、工具调用防重复、循环保护、上下文裁剪、日志和使用假模型的命令行端到端场景。

Windows 子进程树清理测试应在普通本地 PowerShell 或 PyCharm 会话中运行，因为受限执行环境可能会阻止 `taskkill`。

## 退出码

| 退出码 | 含义 |
|---:|---|
| `0` | 模型返回最终回答。 |
| `1` | API 失败、初始化失败或循环保护终止运行。 |
| `2` | 命令行输入非法、工作区无效或任务为空。 |
| `130` | 用户中断运行。 |

