# 上层指令日志（每 20 步重规划一次）

## t=0

- 态势：探索率 0.000、火情发现率 0.000、覆盖率 0.000、人员已知火情 0.000、未合围火点 0 处
- 决策：UAV2 → 中继人员#1(AoI=0, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.00)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.00)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.00)
- 决策：UAV4 → 搜寻象限 [25.0, 25.0](探索率 0.00)
- 决策：UAV5 → 搜寻象限 [25.0, 75.0](探索率 0.00)

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
    },
    {
      "task": "explore",
      "target": [
        25.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    }
  ]
}
```

## t=20

- 态势：探索率 0.225、火情发现率 0.320、覆盖率 1.000、人员已知火情 0.320、未合围火点 2 处
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.23)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.23)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.23)
- 决策：UAV4 → 搜寻象限 [25.0, 25.0](探索率 0.23)
- 决策：UAV5 → 搜寻象限 [25.0, 75.0](探索率 0.23)

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
    },
    {
      "task": "explore",
      "target": [
        25.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    }
  ]
}
```

## t=40

- 态势：探索率 0.360、火情发现率 0.556、覆盖率 0.750、人员已知火情 0.556、未合围火点 4 处
- 决策：UAV2 → 补给（电量 222 < 260）
- 决策：UAV1 → 中继人员#1(AoI=10, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.36)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.36)
- 决策：UAV4 → 搜寻象限 [25.0, 25.0](探索率 0.36)
- 决策：UAV5 → 搜寻象限 [25.0, 75.0](探索率 0.36)

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
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    }
  ]
}
```

## t=60

- 态势：探索率 0.466、火情发现率 0.682、覆盖率 0.312、人员已知火情 0.682、未合围火点 4 处
- 决策：UAV1 → 中继人员#1(AoI=15, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.47)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.47)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.47)
- 决策：UAV4 → 搜寻象限 [25.0, 25.0](探索率 0.47)
- 决策：UAV5 → 搜寻象限 [25.0, 75.0](探索率 0.47)

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
    },
    {
      "task": "explore",
      "target": [
        25.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    }
  ]
}
```

## t=80

- 态势：探索率 0.566、火情发现率 0.816、覆盖率 0.250、人员已知火情 0.816、未合围火点 5 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#1(AoI=35, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.57)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.57)
- 决策：UAV4 → 搜寻象限 [25.0, 25.0](探索率 0.57)
- 决策：UAV5 → 搜寻象限 [25.0, 75.0](探索率 0.57)

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
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    }
  ]
}
```

## t=100

- 态势：探索率 0.631、火情发现率 0.875、覆盖率 0.250、人员已知火情 0.875、未合围火点 5 处
- 决策：UAV2 → 中继人员#1(AoI=55, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.63)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.63)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.63)
- 决策：UAV4 → 搜寻象限 [25.0, 25.0](探索率 0.63)
- 决策：UAV5 → 搜寻象限 [25.0, 75.0](探索率 0.63)

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
    },
    {
      "task": "explore",
      "target": [
        25.0,
        25.0
      ]
    },
    {
      "task": "explore",
      "target": [
        25.0,
        75.0
      ]
    }
  ]
}
```

## t=120

- 态势：探索率 0.708、火情发现率 0.860、覆盖率 0.375、人员已知火情 0.860、未合围火点 5 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#3(AoI=75, covered=False)
- 决策：UAV4 → 中继人员#6(AoI=74, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.71)
- 决策：UAV3 → 巡逻火点 [59.5, 16.9](规模 125)
- 决策：UAV5 → 巡逻火点 [64.7, 78.8](规模 122)

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
      "rescuer_id": 3,
      "handover": true
    },
    {
      "task": "charge"
    },
    {
      "task": "patrol",
      "target": [
        59.5,
        16.9
      ],
      "ring_radius": "auto"
    },
    {
      "task": "relay",
      "rescuer_id": 6,
      "handover": true
    },
    {
      "task": "patrol",
      "target": [
        64.7,
        78.8
      ],
      "ring_radius": "auto"
    }
  ]
}
```

## t=140

- 态势：探索率 0.768、火情发现率 0.819、覆盖率 0.562、人员已知火情 0.819、未合围火点 5 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV4 → 中继人员#5(AoI=94, covered=False)
- 决策：UAV2 → 中继人员#16(AoI=84, covered=False)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.77)
- 决策：UAV5 → 巡逻火点 [59.5, 16.9](规模 125)

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
      "rescuer_id": 16,
      "handover": true
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    },
    {
      "task": "relay",
      "rescuer_id": 5,
      "handover": true
    },
    {
      "task": "patrol",
      "target": [
        59.5,
        16.9
      ],
      "ring_radius": "auto"
    }
  ]
}
```

## t=160

- 态势：探索率 0.823、火情发现率 0.802、覆盖率 0.312、人员已知火情 0.802、未合围火点 5 处
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV4 → 补给（电量 200 < 260）
- 决策：UAV2 → 中继人员#16(AoI=104, covered=False)
- 决策：UAV5 → 中继人员#14(AoI=101, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.82)
- 决策：UAV3 → 巡逻火点 [59.5, 16.9](规模 125)

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
      "rescuer_id": 16,
      "handover": true
    },
    {
      "task": "patrol",
      "target": [
        59.5,
        16.9
      ],
      "ring_radius": "auto"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 14,
      "handover": true
    }
  ]
}
```

## t=180

- 态势：探索率 0.836、火情发现率 0.800、覆盖率 0.562、人员已知火情 0.800、未合围火点 5 处
- 决策：UAV5 → 补给（电量 195 < 260）
- 决策：UAV2 → 补给（电量 198 < 260）
- 决策：UAV1 → 中继人员#16(AoI=124, covered=False)
- 决策：UAV4 → 中继人员#2(AoI=27, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.84)
- 决策：UAV3 → 巡逻火点 [60.0, 17.3](规模 114)

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
      "rescuer_id": 16,
      "handover": true
    },
    {
      "task": "charge"
    },
    {
      "task": "patrol",
      "target": [
        60.0,
        17.3
      ],
      "ring_radius": "auto"
    },
    {
      "task": "relay",
      "rescuer_id": 2,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=200

- 态势：探索率 0.844、火情发现率 0.827、覆盖率 0.500、人员已知火情 0.827、未合围火点 5 处
- 决策：UAV5 → 补给（电量 145 < 260）
- 决策：UAV3 → 补给（电量 152 < 260）
- 决策：UAV1 → 中继人员#2(AoI=47, covered=False)
- 决策：UAV2 → 中继人员#1(AoI=21, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.84)
- 决策：UAV4 → 巡逻火点 [15.5, 30.1](规模 110)

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
      "rescuer_id": 2,
      "handover": true
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
      "task": "patrol",
      "target": [
        15.5,
        30.1
      ],
      "ring_radius": "auto"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=220

- 态势：探索率 0.850、火情发现率 0.805、覆盖率 0.625、人员已知火情 0.805、未合围火点 5 处
- 决策：UAV0 → 补给（电量 148 < 260）
- 决策：UAV1 → 补给（电量 198 < 260）
- 决策：UAV3 → 中继人员#11(AoI=36, covered=False)
- 决策：UAV4 → 中继人员#10(AoI=27, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.85)
- 决策：UAV5 → 巡逻火点 [60.9, 19.7](规模 109)

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
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "relay",
      "rescuer_id": 11,
      "handover": true
    },
    {
      "task": "relay",
      "rescuer_id": 10,
      "handover": true
    },
    {
      "task": "patrol",
      "target": [
        60.9,
        19.7
      ],
      "ring_radius": "auto"
    }
  ]
}
```

## t=240

- 态势：探索率 0.862、火情发现率 0.849、覆盖率 0.625、人员已知火情 0.849、未合围火点 5 处
- 决策：UAV1 → 补给（电量 148 < 260）
- 决策：UAV4 → 补给（电量 148 < 260）
- 决策：UAV3 → 中继人员#10(AoI=47, covered=False)
- 决策：UAV5 → 中继人员#14(AoI=15, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.86)
- 决策：UAV2 → 巡逻火点 [61.2, 18.9](规模 103)

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
      "task": "patrol",
      "target": [
        61.2,
        18.9
      ],
      "ring_radius": "auto"
    },
    {
      "task": "relay",
      "rescuer_id": 10,
      "handover": true
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 14,
      "handover": true
    }
  ]
}
```

## t=260

- 态势：探索率 0.866、火情发现率 0.835、覆盖率 0.312、人员已知火情 0.835、未合围火点 4 处
- 决策：UAV2 → 补给（电量 148 < 260）
- 决策：UAV3 → 补给（电量 198 < 260）
- 决策：UAV4 → 中继人员#10(AoI=67, covered=False)
- 决策：UAV5 → 中继人员#9(AoI=30, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.87)
- 决策：UAV1 → 巡逻火点 [65.1, 80.6](规模 96)

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
      "task": "patrol",
      "target": [
        65.1,
        80.6
      ],
      "ring_radius": "auto"
    },
    {
      "task": "charge"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 10,
      "handover": true
    },
    {
      "task": "relay",
      "rescuer_id": 9,
      "handover": true
    }
  ]
}
```

## t=280

- 态势：探索率 0.869、火情发现率 0.846、覆盖率 0.438、人员已知火情 0.846、未合围火点 4 处
- 决策：UAV5 → 补给（电量 148 < 260）
- 决策：UAV0 → 补给（电量 200 < 260）
- 决策：UAV3 → 中继人员#10(AoI=87, covered=False)
- 决策：UAV4 → 中继人员#9(AoI=50, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.87)
- 决策：UAV2 → 巡逻火点 [65.3, 80.4](规模 92)

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
      "task": "patrol",
      "target": [
        65.3,
        80.4
      ],
      "ring_radius": "auto"
    },
    {
      "task": "relay",
      "rescuer_id": 10,
      "handover": true
    },
    {
      "task": "relay",
      "rescuer_id": 9,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=300

- 态势：探索率 0.880、火情发现率 0.983、覆盖率 0.688、人员已知火情 0.983、未合围火点 3 处
- 决策：UAV4 → 补给（电量 198 < 260）
- 决策：UAV1 → 补给（电量 202 < 260）
- 决策：UAV3 → 中继人员#9(AoI=70, covered=False)
- 决策：UAV5 → 中继人员#3(AoI=53, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.88)
- 决策：UAV2 → 巡逻火点 [22.9, 91.6](规模 94)

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
      "task": "patrol",
      "target": [
        22.9,
        91.6
      ],
      "ring_radius": "auto"
    },
    {
      "task": "relay",
      "rescuer_id": 9,
      "handover": true
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

## t=320

- 态势：探索率 0.885、火情发现率 0.978、覆盖率 0.375、人员已知火情 0.978、未合围火点 3 处
- 决策：UAV4 → 补给（电量 148 < 260）
- 决策：UAV2 → 补给（电量 200 < 260）
- 决策：UAV3 → 中继人员#1(AoI=56, covered=False)
- 决策：UAV0 → 中继人员#10(AoI=20, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.89)
- 决策：UAV5 → 巡逻火点 [22.8, 91.5](规模 92)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 10,
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
      "task": "patrol",
      "target": [
        22.8,
        91.5
      ],
      "ring_radius": "auto"
    }
  ]
}
```

## t=340

- 态势：探索率 0.892、火情发现率 0.990、覆盖率 0.438、人员已知火情 0.990、未合围火点 3 处
- 决策：UAV3 → 补给（电量 150 < 260）
- 决策：UAV5 → 补给（电量 198 < 260）
- 决策：UAV2 → 中继人员#1(AoI=76, covered=False)
- 决策：UAV4 → 中继人员#10(AoI=40, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.89)
- 决策：UAV1 → 巡逻火点 [22.8, 91.5](规模 91)

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
      "task": "patrol",
      "target": [
        22.8,
        91.5
      ],
      "ring_radius": "auto"
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
      "task": "relay",
      "rescuer_id": 10,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=360

- 态势：探索率 0.898、火情发现率 1.000、覆盖率 0.500、人员已知火情 1.000、未合围火点 2 处
- 决策：UAV0 → 补给（电量 150 < 260）
- 决策：UAV1 → 补给（电量 202 < 260）
- 决策：UAV2 → 中继人员#1(AoI=96, covered=False)
- 决策：UAV5 → 中继人员#2(AoI=92, covered=False)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.90)
- 决策：UAV4 → 巡逻火点 [22.8, 91.7](规模 83)

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
    },
    {
      "task": "patrol",
      "target": [
        22.8,
        91.7
      ],
      "ring_radius": "auto"
    },
    {
      "task": "relay",
      "rescuer_id": 2,
      "handover": true
    }
  ]
}
```

## t=380

- 态势：探索率 0.911、火情发现率 1.000、覆盖率 0.188、人员已知火情 1.000、未合围火点 2 处
- 决策：UAV0 → 补给（电量 100 < 260）
- 决策：UAV2 → 补给（电量 198 < 260）
- 决策：UAV5 → 中继人员#2(AoI=112, covered=False)
- 决策：UAV1 → 中继人员#16(AoI=79, covered=False)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.91)
- 决策：UAV4 → 巡逻火点 [22.9, 91.7](规模 78)

```json
{
  "uavs": [
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 16,
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
    },
    {
      "task": "patrol",
      "target": [
        22.9,
        91.7
      ],
      "ring_radius": "auto"
    },
    {
      "task": "relay",
      "rescuer_id": 2,
      "handover": true
    }
  ]
}
```

## t=400

- 态势：探索率 0.914、火情发现率 1.000、覆盖率 0.625、人员已知火情 1.000、未合围火点 2 处
- 决策：UAV2 → 补给（电量 148 < 260）
- 决策：UAV4 → 补给（电量 150 < 260）
- 决策：UAV5 → 中继人员#2(AoI=132, covered=False)
- 决策：UAV0 → 中继人员#1(AoI=16, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.91)
- 决策：UAV3 → 巡逻火点 [22.8, 91.1](规模 62)

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
      "task": "charge"
    },
    {
      "task": "patrol",
      "target": [
        22.8,
        91.1
      ],
      "ring_radius": "auto"
    },
    {
      "task": "charge"
    },
    {
      "task": "relay",
      "rescuer_id": 2,
      "handover": true
    }
  ]
}
```

## t=420

- 态势：探索率 0.925、火情发现率 1.000、覆盖率 0.688、人员已知火情 1.000、未合围火点 1 处
- 决策：UAV3 → 补给（电量 148 < 260）
- 决策：UAV5 → 补给（电量 150 < 260）
- 决策：UAV0 → 中继人员#4(AoI=115, covered=False)
- 决策：UAV4 → 中继人员#1(AoI=36, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.93)
- 决策：UAV2 → 巡逻火点 [91.6, 35.1](规模 94)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 4,
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
      "task": "patrol",
      "target": [
        91.6,
        35.1
      ],
      "ring_radius": "auto"
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

## t=440

- 态势：探索率 0.945、火情发现率 1.000、覆盖率 0.750、人员已知火情 1.000、未合围火点 0 处
- 决策：UAV1 → 补给（电量 152 < 260）
- 决策：UAV0 → 补给（电量 198 < 260）
- 决策：UAV4 → 中继人员#1(AoI=56, covered=False)
- 决策：UAV5 → 中继人员#12(AoI=26, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.94)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.94)

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
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "relay",
      "rescuer_id": 12,
      "handover": true
    }
  ]
}
```

## t=460

- 态势：探索率 0.962、火情发现率 1.000、覆盖率 0.625、人员已知火情 1.000、未合围火点 0 处
- 决策：UAV0 → 补给（电量 148 < 260）
- 决策：UAV2 → 补给（电量 198 < 260）
- 决策：UAV4 → 中继人员#1(AoI=76, covered=False)
- 决策：UAV5 → 中继人员#15(AoI=44, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.96)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.96)

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
      "task": "charge"
    },
    {
      "task": "explore",
      "target": [
        75.0,
        25.0
      ]
    },
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "relay",
      "rescuer_id": 15,
      "handover": true
    }
  ]
}
```

## t=480

- 态势：探索率 0.973、火情发现率 1.000、覆盖率 0.812、人员已知火情 1.000、未合围火点 0 处
- 决策：UAV4 → 补给（电量 150 < 260）
- 决策：UAV3 → 补给（电量 198 < 260）
- 决策：UAV0 → 中继人员#1(AoI=96, covered=False)
- 决策：UAV1 → 中继人员#2(AoI=35, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.97)
- 决策：UAV5 → 搜寻象限 [25.0, 75.0](探索率 0.97)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 1,
      "handover": true
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
        75.0
      ]
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
        25.0,
        75.0
      ]
    }
  ]
}
```

