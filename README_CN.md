<div align="center">

<img src="docs/static/img/RepoMaster.logo.png" alt="RepoMaster Logo" width="200"/>

</div>

# RepoMaster: 基于GitHub仓库的自主任务解决框架

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-red.svg)](paper.pdf)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-orange.svg)](https://github.com/your-org/RepoMaster)

[English](README_EN.md) | [中文](README.md)

</div>

> **📢 重要通知**: 完整的源代码将在论文发表后开源，目前提供的是项目演示和部分代码示例。

## 🎯 快速演示

想象一下，您只需用自然语言描述一个任务，RepoMaster就能自动为您完成后续的一切：从找到最合适的GitHub仓库，到理解其复杂的代码结构，再到最终执行并完成任务！无论是简单的数据提取还是复杂的AI模型应用，RepoMaster都能胜任。

**例如，您可以这样告诉RepoMaster：**

-   **简单任务**: 
    -   "帮我从这个网页上抓取所有的产品名称和价格。"
    -   "提取这份PDF文档中所有的表格数据并保存为CSV文件。"
-   **复杂任务**:
    -   "将这张人物照片转换成梵高油画风格。" (如下方演示)
    -   "自动剪辑这段长视频，提取所有包含特定人物的精彩片段，并添加背景音乐。"
    -   "处理这段会议录音，分离不同发言人的语音，生成会议纪要，并提取关键行动项。"

### 🎨 图像风格迁移任务演示 (复杂任务示例)

<table>
<tr>
<td align="center"><b>原始图像</b></td>
<td align="center"><b>风格参考</b></td>
<td align="center"><b>迁移结果</b></td>
</tr>
<tr>
<td><img src="example/origin.jpg" width="200px" /></td>
<td><img src="example/style.jpg" width="200px" /></td>
<td><img src="example/transfer.jpg" width="200px" /></td>
</tr>
</table>

**RepoMaster 自动化流程 (以风格迁移为例)**:
1. 🔍 **智能搜索**: 自动搜索GitHub上的风格迁移相关仓库。
2. 🏗️ **结构分析**: 分析仓库代码结构，识别核心模型和处理流程。
3. 🔧 **自主执行**: 自动配置环境、加载模型、处理图像。
4. ✅ **完成任务**: 生成风格迁移后的图像，无需人工干预。

### 🎬 完整执行演示 (风格迁移)

观看RepoMaster如何自主完成复杂的图像风格迁移任务：

<div align="center">

<img src="example/demo_ultra_hq.gif" alt="RepoMaster演示" width="800"/>

*RepoMaster自主执行图像风格迁移任务的完整过程*

</div>

**演示亮点**:
- 🤖 **零人工干预**: 从任务理解到结果生成全自动
- 🧠 **智能理解**: 自动理解复杂的深度学习代码库
- ⚡ **高效执行**: 快速定位关键代码，避免无关探索
- 🎯 **精准结果**: 高质量的任务完成效果

---

## 🚀 概述

RepoMaster 是一个革命性的自主代理框架，专门设计用于探索、理解和利用 GitHub 仓库来解决复杂的现实世界任务。与传统的从零开始生成代码的方法不同，RepoMaster 将 GitHub 上的开源仓库视为可组合的工具模块，通过智能搜索、层次化分析和自主探索来自动化地利用这些资源。

### 🎯 核心理念

- **仓库作为工具**：将开源仓库视为解决复杂任务的预制工具包
- **人类化探索**：模拟人类程序员探索未知代码库的方式
- **智能压缩**：在有限的LLM上下文窗口内高效管理海量代码信息
- **自主执行**：端到端地完成从任务理解到代码执行的全流程

## ✨ 主要特性

### 🔍 智能仓库搜索
- 基于任务描述的深度搜索算法
- 多轮查询优化和仓库质量评估
- 自动化的仓库相关性分析

### 🏗️ 层次化仓库分析
- **混合结构建模**：构建层次代码树(HCT)、函数调用图(FCG)和模块依赖图(MDG)
- **核心组件识别**：基于重要性评分自动识别关键模块、类和函数
- **上下文初始化**：智能构建包含README、模块摘要和核心代码的初始上下文

### 🔧 自主探索与执行
- **精细化代码查看**：支持文件、类、函数级别的代码检查
- **依赖分析**：追踪调用链和依赖路径
- **智能搜索**：关键词匹配和语义搜索
- **上下文感知信息选择**：多层次内容压缩策略

### 💡 核心优势
- **高效性**：相比现有框架减少95%的token消耗
- **准确性**：在GitTaskBench上将任务通过率从24.1%提升至62.9%
- **通用性**：支持多种LLM后端(GPT-4o、Claude-3.5、DeepSeek-V3)
- **可扩展性**：模块化设计，支持自定义工具和扩展


## 📊 性能表现

### GitTaskBench 评测结果

| 框架 | LLM | 执行完成率 | 任务通过率 | Token消耗 |
|------|-----|------------|------------|-----------|
| SWE-Agent | Claude 3.5 | 44.44% | 14.81% | 330k |
| OpenHands | Claude 3.5 | 48.15% | 24.07% | 3094k |
| **RepoMaster** | **Claude 3.5** | **75.92%** | **62.96%** | **154k** |

### MLE-R 评测结果

| 框架 | LLM | 有效提交率 | 奖牌获得率 | 金牌率 |
|------|-----|------------|------------|--------|
| SWE-Agent | Claude 3.5 | 50.00% | 4.55% | 4.55% |
| OpenHands | Claude 3.5 | 45.45% | 4.55% | 4.55% |
| **RepoMaster** | **Claude 3.5** | **95.45%** | **27.27%** | **22.73%** |

## 🚀 快速开始

### 安装要求

```bash
# Python 3.11+
pip install -r requirements.txt
```

### 🎯 一句话驱动的自动化任务解决

RepoMaster的核心魅力在于其强大的自主性。您只需提供一个自然语言描述的任务，`RepoMasterAgent` 便会启动其复杂的内部工作流程，为您处理后续的一切。这不仅仅是执行一个预设脚本，而是一个动态的、智能的探索和解决过程。

**当您执行 `repo_master.solve_task_with_repo(task)` 时，RepoMaster 在后台可能执行了如下操作：**

1.  **任务理解与智能搜索**:
    *   解析您的自然语言任务（例如："将这张图转换成梵高风格"）。
    *   自动在GitHub等代码托管平台搜索最相关的开源项目。
    *   评估并筛选出最合适的代码仓库作为解决问题的基础。

2.  **代码深度理解与规划**:
    *   克隆选定的仓库，并对其进行层次化结构分析（阅读README，解析代码结构，识别核心模块、函数等）。
    *   基于任务需求和仓库特性，制定详细的、多步骤的执行计划。

3.  **自主执行与动态适应**:
    *   **环境准备**: 自动处理依赖安装、数据路径配置等。
    *   **代码执行**: 运行从仓库中识别出的关键脚本或生成新的胶水代码。
    *   **智能试错与调试**: 
        *   监控执行过程，捕捉错误和警告 (例如：`NameError`, `FileNotFoundError`, 数值不稳定如 `NaN` loss)。
        *   根据错误信息自主诊断问题原因 (例如：缺失导入、参数不当、文件路径错误)。
        *   动态调整执行策略 (例如：修改脚本、调整超参数、更换优化器、搜索正确的文件名)。
    *   **迭代优化**: 多轮尝试，直到任务成功执行或达到预设的尝试上限。

4.  **结果整合与交付**:
    *   收集执行结果，将其保存到指定位置或格式。
    *   向用户报告任务完成情况和最终产出。

**基本使用示例 (图像风格迁移):**

```python
from core.agent_scheduler import RepoMasterAgent

# 初始化RepoMaster
llm_config = {
    "config_list": [{
        "model": "claude-3-5-sonnet-20241022",
        # "api_key": "your_api_key", # 请替换为您的API Key
        # "base_url": "https://api.anthropic.com" # 请替换为您的API Endpoint
    }],
    "timeout": 2000,
    "temperature": 0.1,
}

code_execution_config = {
    "work_dir": "workspace", 
    "use_docker": False
}

repo_master = RepoMasterAgent(
    llm_config=llm_config,
    code_execution_config=code_execution_config,
)

# 定义一个复杂的AI任务
task = """
我需要将一张内容图片转换成特定艺术风格。
内容图片路径: 'example/origin.jpg'
风格参考图片路径: 'example/style.jpg'
请将最终生成的风格化图片保存为 'workspace/merged_styled_image.png'
"""

# 用户只需一行代码启动任务
# RepoMaster将自动完成搜索、理解、执行、调试的全过程
result_summary = repo_master.solve_task_with_repo(task)

print("任务完成总结:")
print(result_summary)
```

上述代码中的 `task` 字符串是您与RepoMaster交互的核心。RepoMaster会解析这个自然语言任务，并像一个经验丰富的开发者一样，自主地寻找资源、编写或调整代码、处理过程中遇到的问题，最终交付结果。

### 高级用法

#### 1. 直接使用指定仓库

```python
from core.git_task import TaskManager, AgentRunner

# 构造任务配置
task_info = {
    "repo": {
        "type": "github",
        "url": "https://github.com/spatie/pdf-to-text",
    },
    "task_description": "提取PDF文字内容",
    "input_data": [
        {
            "path": "/path/to/input.pdf",
            "description": "要处理的PDF文件"
        }
    ],
}

# 执行任务
result = AgentRunner.run_agent(task_info)
```

#### 2. 本地仓库分析

```python
from core.git_agent import CodeExplorer

# 初始化代码探索器
explorer = CodeExplorer(
    local_repo_path="/path/to/local/repo",
    work_dir="workspace",
    task_type="general",
    use_venv=True,
    llm_config=llm_config
)

# 执行代码分析
task = "分析这个仓库的核心功能并生成使用示例"
result = explorer.code_analysis(task)
```

## 🛠️ 核心组件详解

### 1. 仓库搜索模块 (`deep_research.py`)

```python
async def github_repo_search(self, task):
    """
    执行GitHub仓库深度搜索
    
    参数:
        task: 任务描述
        
    返回:
        匹配仓库的JSON列表
    """
```

### 2. 代码探索工具 (`git_agent.py`)

```python
class CodeExplorer:
    """
    核心代码探索和分析工具
    
    主要功能:
    - 仓库结构分析
    - 依赖关系构建  
    - 智能代码导航
    - 任务驱动的代码生成
    """
```

### 3. 任务管理器 (`git_task.py`)

```python
class TaskManager:
    """
    任务初始化、环境准备和执行管理
    
    主要功能:
    - 工作环境创建
    - 数据集复制和处理
    - 任务配置管理
    """
```

## 📋 配置说明

### LLM配置

```python
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o",  # 支持: gpt-4o, claude-3-5-sonnet, deepseek-chat
            "api_key": "your_api_key",
            "base_url": "api_endpoint"
        }
    ],
    "timeout": 2000,
    "temperature": 0.1,
}
```

### 代码执行配置

```python
code_execution_config = {
    "work_dir": "workspace",      # 工作目录
    "use_docker": False,          # 是否使用Docker
    "timeout": 7200,              # 执行超时时间(秒)
}
```

### 探索工具配置

```python
explorer_config = {
    "max_turns": 40,              # 最大对话轮次
    "use_venv": True,             # 是否使用虚拟环境
    "function_call": True,        # 是否启用函数调用
    "repo_init": True,            # 是否进行仓库初始化分析
}
```

## 🔧 自定义扩展

### 添加自定义工具

```python
from util.toolkits import register_toolkits

def custom_analysis_tool(file_path: str) -> str:
    """自定义分析工具"""
    # 实现你的分析逻辑
    return analysis_result

# 注册工具
register_toolkits(
    [custom_analysis_tool],
    scheduler_agent,
    user_proxy_agent,
)
```

### 扩展仓库搜索

```python
class CustomRepoSearcher:
    def __init__(self):
        self.search_strategies = [
            "keyword_based",
            "semantic_search", 
            "dependency_analysis"
        ]
    
    def search_repositories(self, task_description):
        # 实现自定义搜索逻辑
        pass
```

## 📖 实验与评估

### 复现实验结果

```bash
# 在GitTaskBench上评估
python -m core.git_task --config configs/gittaskbench.yaml

# 在MLE-R上评估  
python -m core.git_task --config configs/mle_r.yaml
```

## 🎬 演示视频

我们提供了详细的演示视频，展示RepoMaster如何:
- 自动搜索和选择相关仓库
- 智能分析复杂代码结构  
- 自主执行多步骤任务
- 处理错误和迭代优化

**🎨 图像风格迁移演示**: 参见[快速演示](#-快速演示)部分，观看RepoMaster如何自主完成复杂的神经网络风格迁移任务。

**🎯 更多演示**: 更多领域的演示视频将随着项目开源陆续发布。

## 📝 使用案例

### 案例1: PDF文字提取

```python
task = """
请提取PDF中第一页的所有文字内容到txt文件中。
输入文件: /path/to/document.pdf
输出要求: 保存为output.txt
"""

result = repo_master.solve_task_with_repo(task)
# RepoMaster会自动:
# 1. 搜索PDF处理相关仓库
# 2. 分析仓库结构和API
# 3. 生成提取代码
# 4. 执行并保存结果
```

### 案例2: 机器学习管道

```python
task = """
基于给定的图像数据集训练一个图像分类模型。
数据集: /path/to/image_dataset/
要求: 使用预训练模型进行微调，保存最佳模型
"""

result = repo_master.solve_task_with_repo(task)
# RepoMaster会自动:
# 1. 找到合适的深度学习仓库
# 2. 理解数据加载和模型结构
# 3. 设置训练管道
# 4. 执行训练并保存模型
```

### 案例3: 视频处理

```python
task = """
从视频中提取关键帧并进行3D姿态估计。
输入: /path/to/video.mp4  
输出: 3D关节点坐标JSON文件
"""

result = repo_master.solve_task_with_repo(task)
# RepoMaster会自动:
# 1. 搜索视频处理和姿态估计仓库
# 2. 理解预处理和推理流程
# 3. 实现端到端处理管道
# 4. 生成结构化输出
```


## 🤝 贡献指南

我们欢迎社区贡献！请参考以下指南：

### 开发环境设置

```bash
git clone https://github.com/your-org/RepoMaster.git
cd RepoMaster
pip install -e ".[dev]"
pre-commit install
```

### 贡献类型

- 🐛 Bug修复
- ✨ 新功能开发
- 📚 文档改进
- 🧪 测试用例添加
- 🔧 工具和实用程序

### 提交流程

1. Fork项目并创建特性分支
2. 编写代码和测试
3. 确保所有测试通过
4. 提交Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📚 引用

如果您在研究中使用了RepoMaster，请引用我们的论文：

```bibtex
@article{repomaster2025,
  title={RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories for Complex Task Solving},
  author={Your Authors},
  journal={NeurIPS},
  year={2025}
}
```

## 🙏 致谢

感谢以下项目和社区的启发和支持：
- [AutoGen](https://github.com/microsoft/autogen) - 多代理对话框架
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) - 软件工程代理平台  
- [SWE-Agent](https://github.com/princeton-nlp/SWE-agent) - GitHub问题解决代理
- [MLE-Bench](https://github.com/openai/mle-bench) - 机器学习工程基准

## 📞 联系我们

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/RepoMaster/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-org/RepoMaster/discussions)

---

<div align="center">

**⭐ 如果 RepoMaster 对您有帮助，请给我们一个星标！**

Made with ❤️ by the RepoMaster Team

</div> 