# Progress

## 项目现状速览

- 用 EfficientNet-B0 做厨具/保温瓶细粒度分类，代码能跑，但**当前模型只训练了 3 类**（`blue_pan`/`pan_package`/`red_pan`），README 宣称的 12 类里另外 9 个保温壶类别还没数据、没训练。
- 跑代码要用装了 torch 的 conda 环境 `clip_verify`（本机路径 `/home/wrz3/miniconda3/envs/clip_verify/bin/python`），系统默认 `python3`（conda base）没装 torch。
- 目录职责：
  - `rawdataset/` —— 训练用原始数据（LabelMe 标注），自备，目前是空的。
  - `classifier_dataset/` —— `prepare_dataset.py` 自动生成，不用手动放。
  - `the_data_need_to_be_testes/` —— 待推理/测试数据，结构同 `rawdataset`，自备，目前是空的（`.gitkeep` 占位）。
- 以上信息已经写进 `README.md`（"0. 准备虚拟环境" + "项目结构" 章节），不用每次都靠这份 PROGRESS.md 转述，README 是最新的。

## predict.py 现有能力

- 支持三种输入：单张图片 / 单个 LabelMe json / 整个目录（结构同 rawdataset）。
- 不传参数时默认扫描 `the_data_need_to_be_testes/`。
- **目录批量模式默认是"安静+汇总"模式**：只打印 JSON 标签和模型预测不一致的项，最后给一份汇总（标明是哪个 session、哪个 json 文件、哪张图），数据量大（几千条）时不用逐行翻。加 `--verbose` 才会像以前一样逐行打印所有结果（含正确的）。
- 没有对"JSON 语法本身写挂（无法解析）"做容错——之前有一次误改成 try/except 处理 JSONDecodeError，被用户否决/未采纳，目前 `json.load` 出错会直接崩溃整个批量任务。如果用户后续要这个健壮性，需要重新加。

## 本次会话（预测输出汇总化）做的事

- 用户反馈批量跑几千条数据时逐行输出没法看，要求"有错误的话最后汇总指出是哪个文件夹哪个文件"。
- 重构了 `predict.py`：抽出 `classify_json()` 做纯计算（不打印），`predict_json()`（单文件详细打印）和 `predict_dir()`（批量，默认只打印不一致项+汇总，`--verbose` 时逐行打印）分别复用它。
- 用合成的假数据（噪声图+labelme json，1 个正确+1 个故意标签写错）跑通验证了默认模式和 `--verbose` 模式，输出格式符合预期，测试完已清理，没有残留文件。
- 同步更新了 `README.md` 的推理命令示例。

## 需要注意的一点（未深入调查，仅记录观察）

- 之前发现的 `/home/wrz3/桌面/rawdataset/20260611_144756/rgb/` 那份真实数据，本次会话再去看时**已经不在了**（`/home/wrz3/桌面/rawdataset/` 现在是空目录），具体是用户自己清理/挪动的，没有细究。
- 顺带看到桌面根目录（`/home/wrz3/桌面/`，不在 `object-classifier/` 项目内）还有一些其他文件：`classifier_dataset/train,val`（已经是裁剪切分好的数据）、`classifier_classes.json`、`verify_labels.py`、`matlab_summary_table.csv`、`yolov10/`、以及桌面根目录自己的一份 `PROGRESS.md`。这些看起来像是同一批物体分类/验证工作的另一套文件，跟 `object-classifier/` 项目是不同位置、可能是不同阶段或不同脚本产出的。**没有去动它们**，只是记录下来，避免以后误以为 `object-classifier/rawdataset/` 空了就等于没有数据可用——用户手头real数据的位置可能已经变了，下次开始前最好先问一句数据现在放哪。

## 本次会话（准备上传 GitHub）做的事

- 用户想把 `object-classifier/` 传到 GitHub。检查发现：本地这个目录**还不是 git 仓库**（没有 `.git`），机器上**没装 `gh` CLI**，`~/.ssh` 下**没有任何 SSH 密钥**，且从没跟 GitHub 做过免密认证。
- 问清楚用户：① GitHub 上还没建仓库，需要新建；② 认证方式选的是 SSH 密钥。
- 已经用 `ssh-keygen -t ed25519 -C "ruizewang27@gmail.com" -f ~/.ssh/id_ed25519 -N ""` 生成了一对新的 SSH 密钥（无密码），私钥在 `~/.ssh/id_ed25519`，公钥在 `~/.ssh/id_ed25519.pub`。

  **公钥内容（用户要贴到 GitHub → Settings → SSH and GPG keys → New SSH key）：**
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKFi5yz6f4+jFrztRmrV8c1QGwOPoV8mEF/sd9Alt6mI ruizewang27@gmail.com
  ```
  （用户反馈在对话里没法直接复制这段文字，所以专门写进这份 PROGRESS.md 方便复制；也可以直接在终端跑 `cat ~/.ssh/id_ed25519.pub` 拿到同样内容。）

- 还需要用户去 https://github.com/new 建一个空仓库（不勾选自动生成 README/.gitignore），把仓库的 SSH 地址（`git@github.com:用户名/仓库名.git`）发回来。
- **卡在等用户回复"公钥加好了 + 仓库建好了 + 仓库地址"这三样**，还没做 `git init`/`commit`/`push`，本地目录目前仍然不是 git 仓库。

## 下次接着做什么

- 如果这次会话是接着上传 GitHub 这件事：先问用户 SSH key 有没有加到 GitHub 账号、仓库建好没有、仓库 SSH 地址是什么（上面公钥内容已经留档，不用重新生成）。拿到地址后再做：
  1. `cd /home/wrz3/桌面/object-classifier && git init`
  2. 用 `git config user.email "ruizewang27@gmail.com"` 和 `git config user.name` 设置本仓库的提交身份（本机全局 git config 之前没读到过，可能没配过，需要现场确认或设置，仓库级配置即可，不用改全局）。
  3. 先跑一次 `ssh -T git@github.com` 确认密钥认证成功（第一次连接可能需要确认 host key）。
  4. `git add` 相关文件（注意 `.gitignore` 已经排除了 `rawdataset/`、`classifier_dataset/`、`the_data_need_to_be_testes/`，`classifier_best.pth` 约 16MB 会正常入库，不需要 Git LFS）、`git commit`、`git remote add origin <SSH地址>`、`git push -u origin main`（或 `master`，注意确认默认分支名）。
- 如果要训练/补齐 12 类：先确认真实标注数据现在具体在哪个路径（上面提到位置可能变了），放进 `object-classifier/rawdataset/<session>/rgb/`，再跑 `prepare_dataset.py` → `train.py`。
- 待推理数据放 `the_data_need_to_be_testes/<session>/rgb/`，直接 `python predict.py`（用 clip_verify 环境）批量跑，默认只看汇总；要看全部逐行结果加 `--verbose`。
- 如果需要对 JSON 语法错误做容错（跳过继续处理而不是整体崩溃），还没做，需要用户明确要求后再加。
- 运行命令统一用：`/home/wrz3/miniconda3/envs/clip_verify/bin/python <script>.py`。
