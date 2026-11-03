# 无人机大小脑协同：当前版本精简代码交付

大脑：OpenAI 兼容接口的大模型高层任务规划，可配置千问等服务；另提供当前版本的启发式高层用于本地对照。小脑：任务条件 MAPPO Actor，接收 explore、patrol、relay、charge 指令，输出9个离散动作。

## 安装与运行

建议 Python 3.10+，在本文件所在目录执行：

```bash
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows 使用 .venv\Scripts\activate
python -m pip install -r requirements.txt
python webapp/server.py
```

打开 http://127.0.0.1:8000/，不要直接双击 index.html。填写自己的 base_url、API key 和模型名后选择“大模型”，点击“开始”。base_url 使用服务商给出的 OpenAI 兼容根地址（如以 /v1 结尾），代码会拼接 /chat/completions。包中不包含任何真实接口密钥。

没有大模型接口时，可选择“启发式”检查环境和底层模型。这里的启发式是当前版本的高层对照，底层仍是强化学习策略。大模型失败会回退到启发式，日志会注明。

## 包含内容

- webapp/：网页、LLM调用、目标完成/失效更新、补给点校验和低电量补给保护、完整环境快照。
- lowlevel/：任务协议、观测、奖励、环境包装、Actor/价值网络、PPO、采样、行为克隆教师、训练入口。目标管理依赖已移至此目录，不依赖原项目的 experiments。
- Env/低层环境.py：当前无人机、火势、人员、通信和补给环境。
- terrain_height.csv：运行所需的地形输入，不是实验结果。
- models/actor_selected.pt：本轮新模型（355维观测、9动作、384隐藏宽度、2层）；模型哈希见 metadata.json。

网页默认加载新模型，确定性寻路辅助默认关闭。目标管理、充电保护调整高层指令，动作由 Actor 输出。可勾选提前结束条件；都关闭时跑满设定步数或由用户停止。

“发送完整环境信息”默认开启：每轮向大模型发送100×100逐格真值及认知图、各机状态、人员、任务进度、地形和近期决策。这是含真值的能力上限测试，单次实测约14万输入tokens。可取消勾选切回简报；点击“查看最近大模型输入”查看实际输入。

## 阅读与训练代码

```bash
python lowlevel/train_lowlevel.py --help
python lowlevel/teacher.py --help
```

训练入口来自当前目标条件PPO代码，重命名并调整了模块路径；训练产生的文件不在包内。Actor权重足够部署；本包没有优化器checkpoint，不能精确恢复原训练进度。当前训练课程的全局/单环境步数同步仍待修正，建议先阅读再长训。

模型是部分验收通过的研究候选：指定区域探索明显改善，但完整绕行和固定坐标驻留仍有不足。不得将“能够运行”理解为各场景均可靠。

## 排除内容

没有初版“高层强化学习＋底层启发式”代码；没有实验数据、runs、曲线、日志、历史权重、完整checkpoint、论文、演示视频、研究报告或虚拟环境。原项目保留不变，此文件夹是独立交付副本。
