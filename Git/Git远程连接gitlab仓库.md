---
Date: 2026-07-16
tags:
  - git
---
## Step1：将本地仓库初始化

```sh
git init
```

## Step2：将本地分支切换为`dev_hlz`

```sh
git branch

git branch -m 当前分支名 dev_hlz
```
![[Git远程连接gitlab仓库-1784172615751.webp]]

## Step3：添加远程仓库

```sh
	git remote add origin git@106.63.4.7:ai-lab/algorithm/endoscope/endogi_visai_server.git

git remote -v
```
- 显示 fetch 和 push 的地址
![[Git远程连接gitlab仓库-1784172710038.webp]]
## Step4：推送本地 `dev_hlz` 到远程并建立跟踪

```sh
git push -u origin dev_hlz
```
- 在远程创建 `dev_hlz` 分支（若不存在）
- 将本地 `dev_hlz` 的所有提交推送到远程
- 设置本地 `dev_hlz` 跟踪远程 `origin/dev_hlz`，之后直接 `git push` 或 `git pull` 即可
![[Git远程连接gitlab仓库-1784172766135.webp]]