# 人员可控：实现与CPU演示

2026-10-26。本轮只实现人员任务控制与前端展示，没有重训、没有使用GPU、没有调用千问。通信继续沿用旧几何覆盖与即时地图交换，没有实现信道/带宽队列。网页默认加载新Actor `research_outputs/lowlevel_training_final/actor_selected.pt`，SHA-256见`verification.json`。

## 如何看效果

查看`session.json`中的URL；本次运行服务位于 `http://127.0.0.1:8000`，已在内置浏览器打开。会话停在t=120等待人员指令。页面回放滑块可以观察此前分散、等待、撤离、连续航点和扑救转场。

- 右侧（窄窗口在地图下方）“人员指挥”输入人员编号、任务和目标，点“下达并继续”。
- 也可以点击地图填写目标，或展开高级JSON填写多个任务及waypoints。
- “沿用任务继续20步”不改任务，只推进一个重规划窗口；普通回放的播放按钮只控制画面，不推进后台仿真。
- 不想逐窗等待时，新开一局前取消“人员逐窗指挥”；后台会连续运行，人工下令可能来不及，演示建议保留该选项。
- `auto`显式释放指定人员，恢复旧启发式；未下达命令的人保持原自动行为。
- 地图显示人员编号、最近轨迹、到当前目标的虚线与作业点。表格显示当前回放帧的任务，状态行同时显示后台和回放时刻，避免混淆。

服务器重启后内存会话失效，可新开一局下达同类指令。`demo_frames.json`保留了本次120步的帧、UAV日志和人员事件，不依赖内存会话作证据；它不是新增的独立对照实验。

## 启动与测试

本机复制来的`.venv/bin/python`没有可执行权限。本轮使用已安装的Python3.13，复用项目site-packages，不修改解释器或重新安装依赖：

```bash
PYTHONPATH=.venv/lib/python3.13/site-packages python3 webapp/server.py --port 8000
PYTHONPATH=.venv/lib/python3.13/site-packages python3 -m unittest discover -s webapp/tests -v
```

其他机器使用其正常Python环境即可。webapp测试共13项通过，其中新增人员测试8项：跨自动重分配保持任务、到达后保持、连续航点与回撤、离线待交付、输入原子验证、过期拒收、恢复自动、无指令兼容旧演化，以及指定区域扑救与灭火（部分合并在同一测试方法）。还检查了HTTP非法编号422、过期窗口409，以及前端继续按钮将后台100步推进至120步。

## Agent接口

```text
POST /api/start
  brain="heuristic", personnel_interactive=true
  steps=240, every=1, seed=730000, stop_when_clear=false

GET /api/personnel/{sid}
  返回时刻、人员状态、人员认知中的火区和最近人员事件

POST /api/personnel/{sid}
  {
    "expected_tick": 40,
    "source": "codex",
    "reason": "依据当前观测的简短安排理由",
    "orders": [
      {"person_ids":[1], "task":"retreat", "target":[5,40]},
      {"person_ids":[2], "task":"hold"},
      {"person_ids":[5], "task":"move", "waypoints":[[18,50],[22,55]]}
    ]
  }
```

人员编号采用环境已有的1至N，不是0至N−1。支持`move`（或`transit`）、`hold`、`retreat`、`contain`、`auto`。`contain`的target为人员已知火区附近的参考坐标，可选`approach_sector=west/east/north/south/any`，由执行器选安全外围作业点。指令默认300步有效，可用`valid_for_ticks`覆盖。移动到位容差0.75格，非精确连续轨迹控制。多条指令先整批校验，再入队。

`orders=[]`表示沿用任务继续。交互模式下每个重规划窗口等待一次显式提交。非交互模式可随时提交，下一物理步消费；若已结束返回409。新指令使用现有`is_covered`（基站或UAV几何连接）判定交付；失联时保留pending，旧任务继续，没有网络带宽/包时延。

## 本次由Codex亲自作出的计划

| 时刻 | 实际决策 |
|---|---|
| 0 | 十人分散到不同接近点，未知火区不盲目分配 |
| 20 | 未收到火情，继续原接近任务 |
| 40 | 全部到达；P1回撤、P2等待、P5两段航点、P6向东转场 |
| 60 | 根据进度让P3、P4、P7继续向不同方向推进 |
| 80 | 已收到东南火区信息，P4/P5/P6分别承担西/北/南侧扑救，当前仍为接近过程 |
| 100 | 通过前端沿用任务继续20步，验证真实按钮链路 |

无人机高层仍是启发式，低层是冻结的新Actor。不能把本次说成“Codex同时控制人机的完整对比实验”，也不能据本次演示断言灭火性能提升。

## 实现文件和限制

- `webapp/personnel_control.py`：校验、待交付/激活状态、任务目标、人员逐帧信息。
- `Env/低层环境.py`：抽出原自动任务更新；仅让未被显式控制的人进入旧分配器，然后复用原移动、局部撤离和灭火。
- `webapp/server.py`：人员控制API、逐窗等待、前端帧、加载新Actor。
- `webapp/static/index.html`：人员控制面板、目标/轨迹、人员编号和窄屏布局。
- `webapp/tests/test_personnel_control.py`：新增行为回归。

有效人员任务不会再被旧“每5步换目标”或全局再平衡覆盖。到达后保持，过期后保持并显示过期，收到auto才交还启发式。紧急本地撤离优先于等待。

本轮保留了旧环境的共享地图和BFS的全图障碍判定假设；近目标安全修正也沿用现有环境判定。这不是已经完成严格局部可观测的信息系统，也不是已完成真实人类自主行为/通信信道模型。不同人员可共享作业位置，未新增人员间碰撞或编队约束。原大型重训方案不是本次执行清单；不要据本演示启动此前暂缓的信道队列改造。
