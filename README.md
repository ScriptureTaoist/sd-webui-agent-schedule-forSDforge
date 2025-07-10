# Agent Scheduler for Stable Diffusion WebUI

Agent Scheduler 是一个为 Stable Diffusion WebUI 设计的扩展，它提供了一个任务队列系统，允许用户安排和管理生成任务。

## 功能

-   **任务队列**: 将多个生成任务添加到一个队列中，并按顺序执行。
-   **历史记录**: 查看已完成、失败或中断的任务。
-   **任务管理**: 暂停、恢复、清除队列，以及重新排队失败的任务。
-   **多检查点支持**: 为每个任务指定不同的检查点。
-   **API 支持**: 通过 API 以编程方式将任务添加到队列中。

## 安装

### SD-Forge

此版本已经过修改以兼容 SD-Forge。（当前处于Bate版本，仍然不可用）

1.  导航到您的 SD-Forge 安装目录。
2.  进入 `extensions` 文件夹。
3.  将此插件的文件夹 (`sd-webui-agent-scheduler-main`) 复制到 `extensions` 目录中。
4.  重启 SD-Forge。

## 使用方法

安装并启用扩展后，您将在 WebUI 的主界面上看到一个新的 "Agent Scheduler" 选项卡。

-   **添加任务**: 在 `txt2img` 或 `img2img` 选项卡中，配置好您的生成参数后，点击 "Enqueue" 按钮将任务添加到队列中。
-   **管理队列**: 在 "Agent Scheduler" -> "Task Queue" 选项卡中，您可以查看待处理的任务，并进行暂停、恢复、清除等操作。
-   **查看历史**: 在 "Agent Scheduler" -> "Task History" 选项卡中，您可以查看已完成的任务及其生成结果。
