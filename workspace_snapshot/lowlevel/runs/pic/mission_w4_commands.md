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

- 态势：探索率 0.140、火情发现率 0.210、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0442、已知火点 1 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.14)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.14)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.14)

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

- 态势：探索率 0.250、火情发现率 0.304、覆盖率 1.000、人员已知火情 0.304、过火面积 0.0454、已知火点 2 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.25)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.25)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.25)

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

- 态势：探索率 0.318、火情发现率 0.433、覆盖率 1.000、人员已知火情 0.433、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.32)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.32)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
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
        25.0
      ]
    }
  ]
}
```

## t=80

- 态势：探索率 0.366、火情发现率 0.422、覆盖率 0.000、人员已知火情 0.418、过火面积 0.0509、已知火点 2 处
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#2(AoI=13, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.37)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.37)

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
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 2,
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

## t=100

- 态势：探索率 0.407、火情发现率 0.571、覆盖率 1.000、人员已知火情 0.571、过火面积 0.0526、已知火点 3 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.41)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.41)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
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
        25.0
      ]
    }
  ]
}
```

## t=120

- 态势：探索率 0.428、火情发现率 0.553、覆盖率 1.000、人员已知火情 0.553、过火面积 0.0541、已知火点 3 处
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.43)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.43)

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

## t=140

- 态势：探索率 0.451、火情发现率 0.516、覆盖率 1.000、人员已知火情 0.516、过火面积 0.0578、已知火点 3 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.45)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
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

- 态势：探索率 0.494、火情发现率 0.521、覆盖率 0.000、人员已知火情 0.521、过火面积 0.0603、已知火点 3 处
- 决策：UAV2 → 补给（电量 200 < 260）
- 决策：UAV3 → 补给（电量 200 < 260）
- 决策：UAV1 → 中继人员#1(AoI=7, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.49)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=180

- 态势：探索率 0.521、火情发现率 0.505、覆盖率 0.000、人员已知火情 0.505、过火面积 0.0616、已知火点 3 处
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=27, covered=False)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.52)

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

- 态势：探索率 0.546、火情发现率 0.501、覆盖率 1.000、人员已知火情 0.501、过火面积 0.0636、已知火点 3 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV3 → 补给（电量 248 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.55)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=220

- 态势：探索率 0.574、火情发现率 0.433、覆盖率 1.000、人员已知火情 0.433、过火面积 0.0671、已知火点 3 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.57)

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

## t=240

- 态势：探索率 0.598、火情发现率 0.400、覆盖率 1.000、人员已知火情 0.400、过火面积 0.0690、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.60)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=260

- 态势：探索率 0.603、火情发现率 0.380、覆盖率 1.000、人员已知火情 0.380、过火面积 0.0707、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.60)

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

## t=280

- 态势：探索率 0.610、火情发现率 0.348、覆盖率 1.000、人员已知火情 0.348、过火面积 0.0735、已知火点 2 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.61)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=300

- 态势：探索率 0.616、火情发现率 0.331、覆盖率 1.000、人员已知火情 0.331、过火面积 0.0755、已知火点 2 处
- 决策：UAV2 → 补给（电量 198 < 260）
- 决策：UAV3 → 补给（电量 200 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.62)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=320

- 态势：探索率 0.644、火情发现率 0.530、覆盖率 1.000、人员已知火情 0.530、过火面积 0.0771、已知火点 3 处
- 决策：UAV0 → 补给（电量 198 < 260）
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.64)

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

## t=340

- 态势：探索率 0.653、火情发现率 0.503、覆盖率 0.000、人员已知火情 0.503、过火面积 0.0797、已知火点 3 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV3 → 补给（电量 252 < 260）
- 决策：UAV1 → 中继人员#1(AoI=11, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.65)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=360

- 态势：探索率 0.663、火情发现率 0.505、覆盖率 1.000、人员已知火情 0.505、过火面积 0.0819、已知火点 3 处
- 决策：UAV2 → 补给（电量 198 < 260）
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.66)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
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

## t=380

- 态势：探索率 0.669、火情发现率 0.453、覆盖率 1.000、人员已知火情 0.453、过火面积 0.0837、已知火点 3 处
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.67)

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
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=400

- 态势：探索率 0.670、火情发现率 0.410、覆盖率 1.000、人员已知火情 0.410、过火面积 0.0847、已知火点 4 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.67)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
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

## t=420

- 态势：探索率 0.674、火情发现率 0.388、覆盖率 1.000、人员已知火情 0.388、过火面积 0.0867、已知火点 2 处
- 决策：UAV0 → 补给（电量 200 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.67)

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

## t=440

- 态势：探索率 0.703、火情发现率 0.376、覆盖率 1.000、人员已知火情 0.376、过火面积 0.0880、已知火点 2 处
- 决策：UAV3 → 补给（电量 200 < 260）
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.70)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=460

- 态势：探索率 0.737、火情发现率 0.360、覆盖率 1.000、人员已知火情 0.360、过火面积 0.0886、已知火点 2 处
- 决策：UAV2 → 补给（电量 198 < 260）
- 决策：UAV0 → 补给（电量 248 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.74)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
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

## t=480

- 态势：探索率 0.778、火情发现率 0.433、覆盖率 1.000、人员已知火情 0.433、过火面积 0.0905、已知火点 3 处
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV3 → 补给（电量 248 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.78)

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
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=500

- 态势：探索率 0.794、火情发现率 0.668、覆盖率 0.500、人员已知火情 0.668、过火面积 0.0922、已知火点 2 处
- 决策：UAV1 → 补给（电量 150 < 260）
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV3 → 中继人员#3(AoI=16, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.79)

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
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 3,
      "handover": true
    }
  ]
}
```

## t=520

- 态势：探索率 0.822、火情发现率 0.648、覆盖率 0.000、人员已知火情 0.648、过火面积 0.0932、已知火点 2 处
- 决策：UAV0 → 补给（电量 200 < 260）
- 决策：UAV3 → 补给（电量 252 < 260）
- 决策：UAV2 → 中继人员#3(AoI=36, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.82)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "relay",
      "rescuer_id": 3,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=540

- 态势：探索率 0.828、火情发现率 0.626、覆盖率 0.000、人员已知火情 0.626、过火面积 0.0945、已知火点 2 处
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV0 → 中继人员#3(AoI=56, covered=False)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.83)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 3,
      "handover": true
    },
    {
      "task": "charge"
    },
    {
      "task": "charge"
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

## t=560

- 态势：探索率 0.835、火情发现率 0.708、覆盖率 0.000、人员已知火情 0.586、过火面积 0.0966、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#3(AoI=76, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.83)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    },
    {
      "task": "relay",
      "rescuer_id": 3,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=580

- 态势：探索率 0.840、火情发现率 0.781、覆盖率 0.000、人员已知火情 0.593、过火面积 0.0975、已知火点 2 处
- 决策：UAV3 → 补给（电量 200 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#3(AoI=96, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.84)

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
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 3,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

