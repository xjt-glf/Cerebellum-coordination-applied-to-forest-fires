# 本地复现
在项目根目录，使用可导入 torch/numpy/fastapi 的 Python 3.13。此工作区的依赖位于 `.venv/lib/python3.13/site-packages`。

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.venv/lib/python3.13/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 experiments/brain_comparison/launch.py
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 experiments/brain_comparison/report.py
```

launch 跳过已有 summary 的完整局；新复现实验请使用独立输出目录或先归档整个本轮目录，不能把旧决策日志追加到新一次执行。
API 凭据仅从已有 webapp.server.RunCfg 在内存读取，输出不保存凭据。不要公开复制含凭据的 server.py。每局 30 次调用，共 300 次逻辑调用，完整格图输入 token 较多。

十路并发 LLM 局，两路并发启发式局，每局独立进程。两组各 10 个固定种子，无筛种，无参数选优。环境不随 API 墙钟等待而演化。transport 重试最多三次，失败按既有实现降级并保留 source。

正式报告生成：运行 `python experiments/brain_comparison/finish_pure.py`。`episodes.csv` 的 planning_seconds 为本地本次执行耗时（含免费重放），historical_planning_seconds 为对应实际 API 调用的历史耗时总和；API 计费口径见 api_accounting.json。
