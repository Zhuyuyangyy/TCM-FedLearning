# TCM-FedLearning

面向多中心中医临床数据的隐私保护联邦学习

## 核心创新点

1. **隐私保护** — 联邦学习框架，数据不出院
2. **多算法支持** — FedAvg / FedProx / SCAFFOLD
3. **非IID处理** — 针对中医数据异质性的优化
4. **证候预测** — 联邦证候辨识模型训练

## 快速开始

```bash
pip install -e .
uvicorn backend.main:app --port 8024 --reload
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/models` | GET | 可用联邦算法列表 |
| `/api/train/start` | POST | 启动联邦训练 |
| `/api/train/status` | GET | 训练状态查询 |

## 支持的算法

| 算法 | 特点 | 参考 |
|------|------|------|
| FedAvg | 基础联邦平均 | McMahan et al., 2017 |
| FedProx | 近端正则化 | Li et al., 2020 |
| SCAFFOLD | 控制变量方差缩减 | Karimireddy et al., 2020 |

## License

MIT
