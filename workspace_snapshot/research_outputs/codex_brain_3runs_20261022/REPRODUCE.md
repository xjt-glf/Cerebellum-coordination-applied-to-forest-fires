# 三局代理大脑实验复现

完整逐局数据在 `codex_730000/`、`codex_730001/`、`codex_730002/`。
所有指令已经固定为 `command_0000.json` 至 `command_0580.json`，不需要再调用任何模型。

在项目根目录，使用当前工作区的 Python 3.13 与既有依赖：

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.venv/lib/python3.13/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 experiments/codex_brain/replay.py 730000
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 experiments/codex_brain/replay.py 730001
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 experiments/codex_brain/replay.py 730002
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 experiments/codex_brain/report.py
```

重放输出单独保存在本目录的 `replay_verification/`，与原始轨迹逐条比较；已存在完整重放则直接验证，不重复追加。
报告还读取上轮正式API与启发式的相同三个种子：`research_outputs/brain_comparison_20261022_recovered/`，不能用上轮10局均值与本轮3局均值混比。

若想开启一次新的交互规划，先为 bridge.py 设置独立输出目录/新种子，并同步control.py的种子列表。bridge每20步保存完整快照并暂停，必须由agent显式提交4架无人机指令；它不会自动生成计划。直接在原目录重复启动会跳过已有summary，不能算新实验。
