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

## 本次会话（上传 GitHub）做的事 —— 已完成

- `object-classifier/` 已经成功传到 GitHub：**https://github.com/WRZ3/object_classifier**（`main` 分支）。
- 过程：本地目录之前不是 git 仓库、机器没装 `gh` CLI、也没有 SSH 密钥。生成了新的 SSH 密钥对（`ssh-keygen -t ed25519 -C "ruizewang27@gmail.com" -f ~/.ssh/id_ed25519 -N ""`，无密码），用户把公钥加到了 GitHub 账号，`ssh -T git@github.com` 验证认证成功（显示 `Hi WRZ3!`）。
- `git init`，仓库级设置 `user.name=WRZ3` / `user.email=ruizewang27@gmail.com`（只设了本仓库的，没碰全局 git config）。
- 用户在 GitHub 上新建仓库时**顺手勾选了自动生成 README**（远程有一个只含 `# object_classifier` 一行的初始提交，跟本地历史不相关）。用户确认后**用 `git push --force` 覆盖**了远程那个占位提交，本地 9 个文件（`.gitignore`/`README.md`/`PROGRESS.md`/`classes.json`/`classifier_best.pth`/`predict.py`/`prepare_dataset.py`/`requirements.txt`/`train.py`）作为唯一的 "Initial commit" 推了上去。
- `rawdataset/`、`classifier_dataset/`、`the_data_need_to_be_testes/` 按 `.gitignore` 正确排除，没有传上去（`classifier_best.pth` 约 16MB，直接进普通 git 历史，没用 Git LFS，目前没问题）。
- 本机现状：`~/.ssh/id_ed25519`（私钥）+ `id_ed25519.pub`（公钥）已生成并配置好，之后这台机器对这个 GitHub 账号的 push/pull 都是免密的，不用再重新配置。

## 本次会话（推送新版本模型）做的事 —— 已完成

- 用户本地已经产出了新版本的 `classifier_best.pth`（16MB → 16MB，字节内容变了）以及配套更新的 `classes.json`、`PROGRESS.md`，要求传到远程仓库 `git@github.com:WRZ3/object_classifier.git`（即上一 session 建的 `origin`）。
- 只 `git add` 了这三个文件（`classifier_best.pth` / `classes.json` / `PROGRESS.md`），提交信息 "Update classifier model to new version"，`git push origin main` 成功（`38fb054..5d88022`）。
- 项目目录下新增了一个未跟踪的 `lingshi/` 目录（含若干 `202606xx_xxxxxx` 命名的子目录，看结构像推理/测试数据的 session 输出，跟 `.gitignore` 里排除的 `rawdataset/`、`the_data_need_to_be_testes/` 性质类似），**这次故意没有加入提交**，也没问用户这个目录具体是什么、要不要入库——下次如果还在，最好确认一下是不是该在 `.gitignore` 里正式排除掉，还是需要传上去。

## 本次会话（新增标注可视化工具，定位一条错误标注）—— 已完成

- 用户反馈 `the_data_need_to_be_testes/20260611_150213/rgb/20260611_150234_109.json` 里 `red_pan` 被预测成 `blue_pan (99.3%)`，但光看 json/图片肉眼看不出具体是哪个标注框错了（这张图里有两个 `red_pan` 标注）。
- 新增 `visualize_annotations.py`（项目根目录）：只依赖 opencv/numpy，不需要 torch，跟能不能跑 predict.py 无关。
  - 用法：`python3 visualize_annotations.py <json路径>`，可选 `--label <某标签>` 只看某一类框，`--out-dir` 改输出位置。
  - 输出：`<basename>_annotated.png`（整图画出所有标注框+序号+标签+score）+ `<basename>_crops/`（每个框单独裁剪成 `序号_标签.png`，方便放大核对）。
- 用它跑了上面那个案例，确认是 **`#01` 号 `red_pan` 标注（bbox 696,282,1028,382，score 0.45）标错了**——裁出来一看实际是蓝色平底锅，是标注问题不是模型问题。`#00` 号 `red_pan`（score 0.81）是真的红锅，没问题。
- 这个工具会在每次跑的时候在 json 同目录下留下 `_annotated.png` 和 `_crops/` 文件夹（不是临时文件，是给用户看的产出物，没有清理）。目前 `the_data_need_to_be_testes/20260611_150213/rgb/` 下已经多了这两样东西。
- 用户在自己的 `clip_verify` 环境里实测跑通了这个脚本（`python3 visualize_annotations.py the_data_need_to_be_testes/20260611_150213/rgb/20260611_150234_109.json`），命令行汇总输出正常，问过裁剪图在哪之后确认是文件管理器打开文件夹能看到，没有更多问题。
- 已把 `visualize_annotations.py` + 更新的 `PROGRESS.md` 提交推送到 GitHub（`git@github.com:WRZ3/object_classifier.git`，`main` 分支，commit `1c383e2`，`5d88022..1c383e2`）。推送前用 AskUserQuestion 确认过范围：只提交这两个文件，`lingshi/` 目录继续保持未跟踪状态不推送（用户选的是"按此范围推送"而不是顺便把 lingshi/ 加进 .gitignore）。

## 下次接着做什么

- 以后本项目改动要同步到 GitHub：正常 `git add` + `git commit` + `git push`（不需要 `--force`，那只是最初为了覆盖占位 README 用的一次性操作，以后除非明确要改写历史否则不要再用）。
- `lingshi/` 目录待确认：是否要正式加进 `.gitignore`（如果是本地临时推理输出），还是需要作为数据/结果的一部分提交上去。
- 如果要训练/补齐 12 类：先确认真实标注数据现在具体在哪个路径（之前发现桌面上的原始位置已经清空/挪动过，位置可能变了），放进 `object-classifier/rawdataset/<session>/rgb/`，再跑 `prepare_dataset.py` → `train.py`。
- 待推理数据放 `the_data_need_to_be_testes/<session>/rgb/`，直接 `python predict.py`（用 clip_verify 环境）批量跑，默认只看汇总；要看全部逐行结果加 `--verbose`。
- 如果需要对 JSON 语法错误做容错（跳过继续处理而不是整体崩溃），还没做，需要用户明确要求后再加。
- 运行命令统一用：`/home/wrz3/miniconda3/envs/clip_verify/bin/python <script>.py`。
- 待确认：`20260611_150234_109.json` 里 `#01` 号 `red_pan` 标注要不要改成 `blue_pan`（已确认是标注错误，见上一节）。已问过用户是先手动改这一条，还是让我批量扫一遍 `the_data_need_to_be_testes/` 找出所有类似的错误标注再一起改——回复还没收到，等用户决定。
