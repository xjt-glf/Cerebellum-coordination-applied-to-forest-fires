# 集中式指令日志（上层每 20 步重规划一次）

## t=0

- 态势：探索率 0.000、火情发现率 0.000、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0422、已知火点 0 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.00)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.00)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.00)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=20

- 态势：探索率 0.059、火情发现率 0.002、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0442、已知火点 0 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.06)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.06)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.06)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=40

- 态势：探索率 0.079、火情发现率 0.002、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0454、已知火点 0 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.08)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.08)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.08)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=60

- 态势：探索率 0.089、火情发现率 0.002、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0485、已知火点 0 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.09)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.09)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.09)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=80

- 态势：探索率 0.102、火情发现率 0.033、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0509、已知火点 1 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.10)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.10)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.10)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=100

- 态势：探索率 0.109、火情发现率 0.061、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0526、已知火点 1 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.11)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.11)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.11)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=120

- 态势：探索率 0.120、火情发现率 0.140、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0541、已知火点 1 处
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.12)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.12)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.12)

```json
{
  "uavs": [
    {
      "task": "explore",
      "target": [
        25.0,
        25.0
      ]
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=140

- 态势：探索率 0.129、火情发现率 0.183、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0578、已知火点 1 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.13)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=160

- 态势：探索率 0.133、火情发现率 0.177、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0603、已知火点 1 处
- 决策：UAV0 → 补给（电量 200 < 260）
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.13)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=180

- 态势：探索率 0.135、火情发现率 0.184、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0616、已知火点 1 处
- 决策：UAV0 → 补给（电量 150 < 260）
- 决策：UAV1 → 补给（电量 150 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.14)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    }
  ]
}
```

## t=200

- 态势：探索率 0.137、火情发现率 0.187、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0636、已知火点 1 处
- 决策：UAV0 → 补给（电量 100 < 260）
- 决策：UAV1 → 补给（电量 100 < 260）
- 决策：UAV2 → 补给（返航余量 -4 已为负）
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    }
  ]
}
```

## t=220

- 态势：探索率 0.153、火情发现率 0.173、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0672、已知火点 1 处
- 决策：UAV0 → 补给（返航余量 -63 已为负）
- 决策：UAV1 → 补给（返航余量 -33 已为负）
- 决策：UAV2 → 补给（返航余量 -28 已为负）
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    }
  ]
}
```

