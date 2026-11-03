# 短航迹实验复现

项目根目录运行；使用项目Python依赖（torch、numpy、scipy、gymnasium、matplotlib等），无需千问密钥、GPU或训练。Actor路径固定为 `research_outputs/lowlevel_training_final/actor_selected.pt`，SHA-256见audit.json。

Mac本次解释器是 `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13`，依赖位于 `.venv/lib/python3.13/site-packages`。其他机器可换成自己的已安装依赖解释器。

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/.venv/lib/python3.13/site-packages"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
python3 experiments/route_brain/test_converter.py
python3 experiments/route_brain/bridge.py 730000 replay
python3 experiments/route_brain/bridge.py 730001 replay
python3 experiments/route_brain/bridge.py 730002 replay
python3 experiments/route_brain/bridge.py 730000 off
python3 experiments/route_brain/bridge.py 730001 off
python3 experiments/route_brain/bridge.py 730002 off
python3 experiments/route_brain/report.py
```

已有summary的运行器会跳过，避免重复追加。要从零验证某个重放，先将该 `replay_<seed>` 或 `off_<seed>` 目录改名留档，再执行对应命令；保留 `codex_<seed>` 在线原始数据。三个种子可用三个独立进程并行，不能共享同一环境在线程中跑。

`replay` 使用保存的90条指令，是确定性核验而非重新调用大脑；新增仪表只读取状态，要求600条完整steps与原始数据逐条相等才写verification。`off` 使用同一批指令，把route降为首航点，不会根据分叉后的状态重新规划。三次off不能统计为三次独立自适应大脑实验。

重新进行自适应实验需另设输出目录和新种子，启动 `bridge.py <seed>`，由大脑读取waiting.json后通过control.submit逐窗口提交计划；不得自动用过去计划冒充新的在线思考。原始command包含当时的规划理由。

转化器使用方式：已有环境与公共目标管理器构造 `ShortRouteConverter(env, goals)`；每20步以apply接收四机计划；每物理步先执行公共充电保护再tick，然后Actor选动作；第20步后调用finish_window记完成或到期。例子：

```json
{"task":"explore","route":[[62,45],[62,52],[56,57]]}
```

这是UAV当前位于(68,39)时的19步理想航程折线。直线则使用 `{"task":"explore","target":[56,15]}`。只有route会自动切内部点，单航点沿用原公共目标更新规则。充电抢占后队列不会恢复；下一窗口须重新给计划。

报告依赖旧版单航点的 `research_outputs/comprehensive_comparison_3seeds_20261022/analysis.json` 和原3局summary，用于逐种子比较与初态/权重哈希核验。本轮新增API调用为0；交互等待时间不作为裸模型推理延迟指标。
