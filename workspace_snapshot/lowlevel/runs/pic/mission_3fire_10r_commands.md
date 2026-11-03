# 集中式指令日志（上层每 20 步重规划一次）

## t=0

- 态势：探索率 0.000、火情发现率 0.000、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0319、已知火点 0 处
- 决策：UAV3 → 中继人员#1(AoI=0, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.00)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.00)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.00)

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

## t=20

- 态势：探索率 0.144、火情发现率 0.160、覆盖率 0.000、人员已知火情 0.000、过火面积 0.0338、已知火点 1 处
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.14)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.14)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.14)

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

## t=40

- 态势：探索率 0.262、火情发现率 0.153、覆盖率 1.000、人员已知火情 0.153、过火面积 0.0357、已知火点 1 处
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.26)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.26)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.26)

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

## t=60

- 态势：探索率 0.343、火情发现率 0.774、覆盖率 1.000、人员已知火情 0.774、过火面积 0.0377、已知火点 3 处
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.34)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.34)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.34)

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

## t=80

- 态势：探索率 0.408、火情发现率 0.753、覆盖率 0.700、人员已知火情 0.753、过火面积 0.0385、已知火点 3 处
- 决策：UAV3 → 中继人员#1(AoI=0, covered=True)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.41)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.41)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.41)

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

## t=100

- 态势：探索率 0.459、火情发现率 0.696、覆盖率 0.800、人员已知火情 0.696、过火面积 0.0403、已知火点 3 处
- 决策：UAV1 → 中继人员#1(AoI=9, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.46)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.46)
- 决策：UAV3 → 搜寻象限 [75.0, 25.0](探索率 0.46)

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

## t=120

- 态势：探索率 0.523、火情发现率 0.678、覆盖率 0.500、人员已知火情 0.678、过火面积 0.0417、已知火点 3 处
- 决策：UAV3 → 中继人员#8(AoI=3, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.52)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.52)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.52)

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
      "task": "explore",
      "target": [
        75.0,
        75.0
      ]
    },
    {
      "task": "relay",
      "rescuer_id": 8,
      "handover": true
    }
  ]
}
```

## t=140

- 态势：探索率 0.569、火情发现率 0.640、覆盖率 0.600、人员已知火情 0.640、过火面积 0.0432、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#2(AoI=8, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.57)

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
      "rescuer_id": 2,
      "handover": true
    }
  ]
}
```

## t=160

- 态势：探索率 0.630、火情发现率 0.662、覆盖率 0.400、人员已知火情 0.662、过火面积 0.0442、已知火点 2 处
- 决策：UAV2 → 补给（电量 200 < 260）
- 决策：UAV3 → 补给（电量 200 < 260）
- 决策：UAV1 → 中继人员#2(AoI=28, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.63)

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

## t=180

- 态势：探索率 0.676、火情发现率 0.654、覆盖率 0.500、人员已知火情 0.654、过火面积 0.0450、已知火点 2 处
- 决策：UAV0 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 248 < 260）
- 决策：UAV3 → 中继人员#7(AoI=13, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.68)

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
      "rescuer_id": 7,
      "handover": true
    }
  ]
}
```

## t=200

- 态势：探索率 0.700、火情发现率 0.617、覆盖率 0.500、人员已知火情 0.617、过火面积 0.0454、已知火点 2 处
- 决策：UAV3 → 补给（电量 248 < 260）
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#3(AoI=18, covered=False)
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
      "rescuer_id": 3,
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

- 态势：探索率 0.727、火情发现率 0.587、覆盖率 0.300、人员已知火情 0.587、过火面积 0.0464、已知火点 2 处
- 决策：UAV0 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#5(AoI=18, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.73)

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
      "rescuer_id": 5,
      "handover": true
    }
  ]
}
```

## t=240

- 态势：探索率 0.761、火情发现率 0.654、覆盖率 0.600、人员已知火情 0.654、过火面积 0.0471、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#3(AoI=13, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.76)

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
      "task": "charge"
    }
  ]
}
```

## t=260

- 态势：探索率 0.780、火情发现率 0.728、覆盖率 0.500、人员已知火情 0.728、过火面积 0.0478、已知火点 2 处
- 决策：UAV0 → 补给（电量 248 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=20, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.78)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

## t=280

- 态势：探索率 0.785、火情发现率 0.857、覆盖率 0.700、人员已知火情 0.857、过火面积 0.0481、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV0 → 中继人员#5(AoI=40, covered=False)
- 决策：UAV1 → 搜寻象限 [25.0, 75.0](探索率 0.79)

```json
{
  "uavs": [
    {
      "task": "relay",
      "rescuer_id": 5,
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
      "task": "charge"
    }
  ]
}
```

## t=300

- 态势：探索率 0.789、火情发现率 0.871、覆盖率 0.300、人员已知火情 0.871、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#5(AoI=60, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.79)

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
      "rescuer_id": 5,
      "handover": true
    }
  ]
}
```

## t=320

- 态势：探索率 0.792、火情发现率 0.874、覆盖率 0.500、人员已知火情 0.874、过火面积 0.0485、已知火点 2 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV3 → 补给（电量 252 < 260）
- 决策：UAV1 → 中继人员#9(AoI=42, covered=False)
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
      "rescuer_id": 9,
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

- 态势：探索率 0.792、火情发现率 0.865、覆盖率 0.500、人员已知火情 0.865、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=9, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.79)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

## t=360

- 态势：探索率 0.792、火情发现率 0.855、覆盖率 0.500、人员已知火情 0.855、过火面积 0.0485、已知火点 2 处
- 决策：UAV2 → 补给（电量 248 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#9(AoI=18, covered=False)
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
      "rescuer_id": 9,
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

## t=380

- 态势：探索率 0.794、火情发现率 0.844、覆盖率 0.500、人员已知火情 0.844、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=7, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.79)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

## t=400

- 态势：探索率 0.797、火情发现率 0.830、覆盖率 0.500、人员已知火情 0.830、过火面积 0.0485、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 252 < 260）
- 决策：UAV1 → 中继人员#9(AoI=18, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.80)

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
      "rescuer_id": 9,
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

## t=420

- 态势：探索率 0.797、火情发现率 0.819、覆盖率 0.500、人员已知火情 0.819、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=8, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.80)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

## t=440

- 态势：探索率 0.802、火情发现率 0.790、覆盖率 0.500、人员已知火情 0.790、过火面积 0.0485、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#9(AoI=18, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.80)

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
      "rescuer_id": 9,
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

- 态势：探索率 0.802、火情发现率 0.750、覆盖率 0.500、人员已知火情 0.750、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=7, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.80)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

## t=480

- 态势：探索率 0.807、火情发现率 0.721、覆盖率 0.500、人员已知火情 0.721、过火面积 0.0485、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 252 < 260）
- 决策：UAV1 → 中继人员#9(AoI=18, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.81)

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
      "rescuer_id": 9,
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

## t=500

- 态势：探索率 0.807、火情发现率 0.704、覆盖率 0.500、人员已知火情 0.704、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=8, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.81)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

## t=520

- 态势：探索率 0.809、火情发现率 0.660、覆盖率 0.500、人员已知火情 0.660、过火面积 0.0485、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 250 < 260）
- 决策：UAV1 → 中继人员#9(AoI=18, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.81)

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
      "rescuer_id": 9,
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

## t=540

- 态势：探索率 0.809、火情发现率 0.714、覆盖率 0.500、人员已知火情 0.714、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=7, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.81)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

## t=560

- 态势：探索率 0.810、火情发现率 0.690、覆盖率 0.500、人员已知火情 0.690、过火面积 0.0485、已知火点 2 处
- 决策：UAV2 → 补给（电量 250 < 260）
- 决策：UAV3 → 补给（电量 252 < 260）
- 决策：UAV1 → 中继人员#9(AoI=18, covered=False)
- 决策：UAV0 → 搜寻象限 [25.0, 25.0](探索率 0.81)

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
      "rescuer_id": 9,
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

## t=580

- 态势：探索率 0.810、火情发现率 0.640、覆盖率 0.500、人员已知火情 0.640、过火面积 0.0485、已知火点 2 处
- 决策：UAV0 → 补给（电量 250 < 260）
- 决策：UAV1 → 补给（电量 250 < 260）
- 决策：UAV3 → 中继人员#4(AoI=8, covered=False)
- 决策：UAV2 → 搜寻象限 [75.0, 75.0](探索率 0.81)

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
      "rescuer_id": 4,
      "handover": true
    }
  ]
}
```

