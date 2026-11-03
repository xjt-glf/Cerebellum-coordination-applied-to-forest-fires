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

- 态势：探索率 0.144、火情发现率 0.251、覆盖率 1.000、人员已知火情 0.251、过火面积 0.0442、已知火点 1 处
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

- 态势：探索率 0.236、火情发现率 0.251、覆盖率 1.000、人员已知火情 0.251、过火面积 0.0454、已知火点 1 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.24)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.24)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.24)

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

- 态势：探索率 0.324、火情发现率 0.377、覆盖率 0.875、人员已知火情 0.377、过火面积 0.0485、已知火点 3 处
- 决策：UAV0 → 中继人员#6(AoI=8, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.32)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.32)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.32)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 6,
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

- 态势：探索率 0.430、火情发现率 0.499、覆盖率 0.125、人员已知火情 0.499、过火面积 0.0506、已知火点 3 处
- 决策：UAV0 → 中继人员#2(AoI=3, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.43)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.43)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.43)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 2,
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

- 态势：探索率 0.520、火情发现率 0.355、覆盖率 1.000、人员已知火情 0.355、过火面积 0.0520、已知火点 3 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.52)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.52)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.52)

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

- 态势：探索率 0.577、火情发现率 0.455、覆盖率 1.000、人员已知火情 0.455、过火面积 0.0531、已知火点 3 处
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.58)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.58)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.58)

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

## t=140

- 态势：探索率 0.603、火情发现率 0.611、覆盖率 1.000、人员已知火情 0.611、过火面积 0.0550、已知火点 3 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.60)

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
      "rescuer_id": 1,
      "handover": true
    }
  ]
}
```

## t=160

- 态势：探索率 0.615、火情发现率 0.610、覆盖率 0.000、人员已知火情 0.598、过火面积 0.0563、已知火点 2 处
- 决策：UAV0 → 补给（电量 200 < 260）
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV3 → 中继人员#3(AoI=15, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.61)

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
      "rescuer_id": 3,
      "handover": true
    }
  ]
}
```

## t=180

- 态势：探索率 0.657、火情发现率 0.582、覆盖率 1.000、人员已知火情 0.582、过火面积 0.0570、已知火点 2 处
- 决策：UAV2 → 补给（电量 150 < 260）
- 决策：UAV3 → 补给（电量 150 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.66)

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

## t=200

- 态势：探索率 0.685、火情发现率 0.346、覆盖率 1.000、人员已知火情 0.346、过火面积 0.0579、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.69)

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
      "rescuer_id": 1,
      "handover": true
    }
  ]
}
```

## t=220

- 态势：探索率 0.690、火情发现率 0.338、覆盖率 0.875、人员已知火情 0.338、过火面积 0.0583、已知火点 1 处
- 决策：UAV3 → 补给（电量 248 < 260）
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#2(AoI=7, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.69)

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
      "task": "charge"
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=240

- 态势：探索率 0.711、火情发现率 0.455、覆盖率 0.125、人员已知火情 0.455、过火面积 0.0591、已知火点 2 处
- 决策：UAV3 → 补给（电量 198 < 260）
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#2(AoI=27, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.71)

```json
{
  "uavs": [
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
        75.0
      ]
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=260

- 态势：探索率 0.732、火情发现率 0.738、覆盖率 0.000、人员已知火情 0.455、过火面积 0.0594、已知火点 1 处
- 决策：UAV1 → 补给（电量 200 < 260）
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV0 → 中继人员#2(AoI=47, covered=False)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.73)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 2,
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

## t=280

- 态势：探索率 0.783、火情发现率 1.000、覆盖率 0.625、人员已知火情 1.000、过火面积 0.0604、已知火点 1 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=55, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.78)

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
      "task": "charge"
    }
  ]
}
```

## t=300

- 态势：探索率 0.789、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV0 → 补给（电量 200 < 260）
- 决策：UAV1 → 补给（电量 248 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.79)

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

## t=320

- 态势：探索率 0.789、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV2 → 补给（电量 200 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#1(AoI=0, covered=True)
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

## t=340

- 态势：探索率 0.793、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV1 → 补给（电量 248 < 260）
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.79)

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

## t=360

- 态势：探索率 0.793、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV1 → 补给（电量 198 < 260）
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.79)

```json
{
  "uavs": [
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

- 态势：探索率 0.812、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV3 → 补给（电量 200 < 260）
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.81)

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
      "task": "charge"
    }
  ]
}
```

## t=400

- 态势：探索率 0.816、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 252 < 260）
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.82)

```json
{
  "uavs": [
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

- 态势：探索率 0.817、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
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

- 态势：探索率 0.818、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 252 < 260）
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.82)

```json
{
  "uavs": [
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

## t=460

- 态势：探索率 0.818、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
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
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=480

- 态势：探索率 0.819、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 252 < 260）
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.82)

```json
{
  "uavs": [
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

## t=500

- 态势：探索率 0.819、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
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
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=520

- 态势：探索率 0.819、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 252 < 260）
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.82)

```json
{
  "uavs": [
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

## t=540

- 态势：探索率 0.819、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
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
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

## t=560

- 态势：探索率 0.819、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 252 < 260）
- 决策：UAV0 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.82)

```json
{
  "uavs": [
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

## t=580

- 态势：探索率 0.819、火情发现率 1.000、覆盖率 1.000、人员已知火情 1.000、过火面积 0.0604、已知火点 0 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV2 → 中继人员#1(AoI=0, covered=True)
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
      "rescuer_id": 1,
      "handover": true
    },
    {
      "task": "charge"
    }
  ]
}
```

