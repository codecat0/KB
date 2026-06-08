---
Date: 2026-06-08
tags:
  - git
---
### 1. 查看 Git 自身的 HTTP/HTTPS 代理配置
```sh
git config --global --get http.proxy
git config --global --get https.proxy
```

### 2. 配置 Git HTTP/HTTPS 代理
```sh
# 设置代理（使用混合端口 7890）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 验证是否设置成功
git config --global --get http.proxy
git config --global --get https.proxy
```

### 3. 取消代理
```sh
git config --global --unset http.proxy
git config --global --unset https.proxy
```