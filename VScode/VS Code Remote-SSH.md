---
Date: 2026-06-06
tags:
  - VScode
---
## [server] Found running server

```
[server] Found running server (pid=229725)
```
日志显示 `[server] Found running server (pid=229725)`，说明 VS Code 在远程服务器上找到了一个正在运行的 VS Code Server 进程（PID 229725）。但是，后续没有连接成功的日志，直接中断了。 这通常意味着：**远程服务器上的 VS Code Server 进程已经“假死”、状态残留，或者由于网络/SSH隧道波动导致本地客户端无法与该进程进行正常的内部通信。** 结合前面日志出现的 `Existing exec server ... timed out`，进一步印证了远程服务端状态异常。

既然远程进程已经假死，我们需要手动杀掉它并清理缓存，让 VS Code 重新初始化。

执行以下命令杀掉残留进程：
```
pkill -f vscode-server 
```
