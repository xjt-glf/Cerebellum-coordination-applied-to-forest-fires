# 最终交付状态

2026-11-03 21:53:00.185398

真实训练、评测和剩余差异见 final_report.md 与 final_acceptance.json。整体 provisional，不宣称收敛或通过全部门槛。

[
  {
    "seed": 1103,
    "episodes": 1616,
    "updates": 101,
    "pid": 1732,
    "running": false,
    "checkpoint": "/workspace/hrl_dqn_improve/research_outputs/local_allocator_mappo_20261103/v1_count_s1103/checkpoint_01616.pt",
    "sha256": "c511f657b112f3665ab2cfc33d992596d549f28716a18421c826c9530642ef4f",
    "fleet_counts": {
      "3": 404,
      "4": 404,
      "5": 404,
      "6": 404
    },
    "physical_steps": 969600,
    "team_transitions": 48480,
    "agent_samples": 216966
  },
  {
    "seed": 1104,
    "episodes": 1536,
    "updates": 96,
    "pid": 1733,
    "running": false,
    "checkpoint": "/workspace/hrl_dqn_improve/research_outputs/local_allocator_mappo_20261103/v1_count_s1104/checkpoint_01536.pt",
    "sha256": "b2cb05543572e75d43964958f284daec25bc24376192baeca463dbf9d1416d9e",
    "fleet_counts": {
      "3": 384,
      "4": 384,
      "5": 384,
      "6": 384
    },
    "physical_steps": 921600,
    "team_transitions": 46080,
    "agent_samples": 206302
  },
  {
    "seed": 1105,
    "episodes": 1376,
    "updates": 86,
    "pid": 1734,
    "running": false,
    "checkpoint": "/workspace/hrl_dqn_improve/research_outputs/local_allocator_mappo_20261103/v1_count_s1105/checkpoint_01376.pt",
    "sha256": "08e400a8cb2a87a4f6b6b0d5fbd24062ddc4eb9d8b03c493c57cd87480ee0058",
    "fleet_counts": {
      "3": 344,
      "4": 344,
      "5": 344,
      "6": 344
    },
    "physical_steps": 825600,
    "team_transitions": 41280,
    "agent_samples": 184977
  }
]

低层actor SHA256 dfa5cdd0f4b673567c04220af9d82ee55f16920d55398a02471a0d535d59982e


v3 独立分支结果已写入 v3_report.md / v3_final_acceptance.json；原版完整结果见 final_report.md。不据留出回选版本。
