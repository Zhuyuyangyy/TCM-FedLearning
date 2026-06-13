# TCM-FedLearning

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Federated Learning](https://img.shields.io/badge/method-FedAvg%20%7C%20FedProx%20%7C%20SCAFFOLD-blue.svg)

**A privacy-preserving federated learning platform for multi-center Traditional Chinese Medicine (TCM) clinical data, enabling collaborative model training without sharing raw patient data.**

This platform implements three state-of-the-art federated learning algorithms (FedAvg, FedProx, SCAFFOLD) with differential privacy mechanisms, designed specifically for the heterogeneous data distributions found across TCM clinical centers.

---

## Overview

Multi-center TCM clinical data is inherently heterogeneous -- different hospitals see different patient populations, use different diagnostic patterns, and have varying syndrome distributions. Sharing raw patient data across institutions raises significant privacy concerns under regulations like China's Personal Information Protection Law (PIPL).

This platform solves both challenges through federated learning:

1. **Privacy-Preserving Training** -- Models are trained locally at each hospital; only model parameters (not patient data) are aggregated centrally
2. **Non-IID Data Handling** -- FedProx and SCAFFOLD algorithms specifically address the non-independent and identically distributed (non-IID) nature of multi-center TCM data
3. **Differential Privacy** -- Gaussian noise injection provides mathematically guaranteed privacy bounds (epsilon-delta privacy)

---

## Key Features

- **Three Federated Algorithms** -- FedAvg (baseline), FedProx (proximal regularization for non-IID), SCAFFOLD (variance reduction via control variates)
- **Differential Privacy** -- Gaussian mechanism with configurable epsilon-delta privacy budget
- **Non-IID Data Simulation** -- Dirichlet-based partitioning with configurable heterogeneity (alpha parameter)
- **Background Training Jobs** -- Asynchronous training via FastAPI background tasks with real-time status monitoring
- **Synthetic Data Generation** -- Built-in synthetic TCM diagnostic data with per-client distribution shifts
- **Multi-Task Architecture** -- Feed-forward network for TCM syndrome classification tasks
- **RESTful API** -- FastAPI service with training job management and status monitoring

---

## Architecture

```
    Hospital A          Hospital B          Hospital C
    (Local Data)        (Local Data)        (Local Data)
        |                   |                   |
    +---v---+           +---v---+           +---v---+
    | Local |           | Local |           | Local |
    | Model |           | Model |           | Model |
    | Train |           | Train |           | Train |
    +---+---+           +---+---+           +---+---+
        |                   |                   |
        |    DP Noise       |    DP Noise       |    DP Noise
        |   (Optional)      |   (Optional)      |   (Optional)
        |                   |                   |
        +-------------------+-------------------+
                            |
                  +---------v---------+
                  |  Central Server   |
                  |  Aggregation      |
                  |  (FedAvg/FedProx/ |
                  |   SCAFFOLD)       |
                  +---------+---------+
                            |
                  +---------v---------+
                  |  Updated Global   |
                  |  Model            |
                  +-------------------+
                            |
              +-------------v-------------+
              |  Distribute to Clients    |
              +---------------------------+
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| API Framework | FastAPI + Uvicorn |
| Deep Learning | PyTorch 2.0+ |
| Scientific Computing | NumPy, SciPy, scikit-learn |
| Data Processing | Pandas |
| Graph Analysis | NetworkX |
| Configuration | PyYAML, Pydantic |
| Testing | pytest |
| Linting | Ruff |
| CI/CD | GitHub Actions |

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TCM-FedLearning.git
cd TCM-FedLearning

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install fastapi uvicorn scikit-learn numpy pandas pyyaml pydantic pydantic-settings torch scipy networkx
```

### 2. Start the API Server

```bash
# Development server with auto-reload
uvicorn backend.main:app --host 0.0.0.0 --port 8024 --reload
```

Swagger UI is available at: **http://localhost:8024/docs**

### 3. Start a Federated Training Job

```bash
# Via API
curl -X POST http://localhost:8024/api/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "num_clients": 3,
    "global_rounds": 10,
    "local_epochs": 5,
    "algorithm": "fedprox",
    "mu": 0.01,
    "dp_epsilon": 16.0
  }'
```

### 4. Monitor Training Status

```bash
curl http://localhost:8024/api/train/status
```

### 5. Run the Experiment Pipeline

```bash
python scripts/run_experiment.py
```

This runs FedAvg aggregation correctness tests, non-IID partition analysis, and privacy budget computation. Results are saved to `output/fed_learning_results.json`.

### 6. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run backend tests
pytest backend/tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend
```

---

## Supported Algorithms

| Algorithm | Reference | Key Mechanism | Best For |
|-----------|-----------|---------------|----------|
| **FedAvg** | McMahan et al., 2017 | Simple weighted averaging of client models | IID data, baseline |
| **FedProx** | Li et al., 2020 | Proximal term (mu/2)\|\|w - w_global\|\|^2 penalizes client drift | Non-IID data, heterogeneous networks |
| **SCAFFOLD** | Karimireddy et al., 2020 | Control variates for variance reduction, correcting client drift | High heterogeneity, faster convergence |

### FedProx

Adds a proximal term to the local loss function:

```
L_local = L_task + (mu / 2) * ||w_local - w_global||^2
```

The coefficient `mu` controls coupling strength. `mu=0` recovers FedAvg; typical range [0.001, 0.1].

### SCAFFOLD

Maintains server control variate `c` and client control variates `c_i`. The SGD update becomes:

```
w <- w - lr * (g + c - c_i)
```

This steers local updates toward the global objective, reducing variance from client drift.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/models` | GET | List available federated algorithms |
| `/api/train/start` | POST | Start a federated training job |
| `/api/train/status` | GET | Query training job status |

### API Usage Example

```python
import httpx

# Start a FedProx training job with differential privacy
response = httpx.post("http://localhost:8024/api/train/start", json={
    "num_clients": 5,
    "global_rounds": 20,
    "local_epochs": 5,
    "lr": 0.01,
    "batch_size": 32,
    "algorithm": "fedprox",
    "mu": 0.01,
    "dp_epsilon": 8.0,
    "dp_delta": 1e-5,
    "seed": 42
})
print(response.json())

# Check training status
response = httpx.get("http://localhost:8024/api/train/status")
print(response.json())
```

### Training Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_clients` | int | 3 | Number of federated clients (1-100) |
| `global_rounds` | int | 5 | Number of global aggregation rounds (1-1000) |
| `local_epochs` | int | 5 | Local training epochs per client per round (1-100) |
| `lr` | float | 0.01 | Client learning rate |
| `batch_size` | int | 32 | Client mini-batch size |
| `algorithm` | str | "fedavg" | Aggregation algorithm: fedavg / fedprox / scaffold |
| `mu` | float | 0.01 | Proximal coefficient (FedProx only) |
| `dp_epsilon` | float | 16.0 | Differential privacy epsilon (0 = no DP) |
| `dp_delta` | float | 1e-5 | Differential privacy delta |
| `server_lr` | float | 1.0 | Server learning rate (SCAFFOLD only) |
| `seed` | int | 42 | Random seed for reproducibility |

---

## Project Structure

```
TCM-FedLearning/
├── backend/
│   ├── api/
│   │   ├── train.py               # Training job API (start, status)
│   │   └── __init__.py
│   ├── models/
│   │   ├── fedavg.py              # Federated Averaging implementation
│   │   ├── fedprox.py             # FedProx with proximal regularization
│   │   ├── scaffold.py            # SCAFFOLD with control variates
│   │   ├── dp_mechanism.py        # Differential privacy (Gaussian mechanism)
│   │   └── __init__.py
│   ├── config.py                  # Application settings
│   ├── main.py                    # FastAPI application entry point
│   └── __init__.py
├── data/
│   └── federation_config.yaml     # Federation node configuration
├── docs/
│   ├── Claim_Evidence_Table.md    # Research claim-evidence mapping
│   ├── SCI_Paper_Skeleton.md      # Paper draft structure
│   └── 技术交底书.md              # Technical disclosure document
├── scripts/
│   ├── run_experiment.py          # Full experiment pipeline
│   └── quick_smoke_test.py        # Quick validation script
├── tests/
│   └── test_smoke.py              # Smoke tests
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI pipeline
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Dependencies
├── REPRODUCE.md                   # Reproduction guide
└── README.md                      # This file
```

---

## Benchmarks

Run the built-in experiment pipeline to reproduce benchmarks:

```bash
python scripts/run_experiment.py
```

| Experiment | Details |
|-----------|---------|
| FedAvg aggregation | 5 clients, verifies mean-aggregation correctness (tolerance 1e-6) |
| Non-IID partition | Dirichlet alpha sweep: 0.1 (high heterogeneity), 0.5, 1.0, 10.0 (near IID) |
| Privacy budget | Epsilon computation for 10, 50, 100, 200 rounds at sample_rate=0.1 |
| Synthetic data | Per-client distribution shift (client_id * 0.3 offset) |

---

## Research

This platform supports research in privacy-preserving clinical AI and federated learning for healthcare. Related documentation:

- **SCI Paper Skeleton** (`docs/SCI_Paper_Skeleton.md`) -- Draft structure for peer-reviewed publication
- **Claim-Evidence Table** (`docs/Claim_Evidence_Table.md`) -- Mapping of research claims to supporting evidence
- **Technical Disclosure** (`docs/技术交底书.md`) -- Detailed technical methodology

### Citation

```bibtex
@software{tcm_fedlearning,
  title={TCM-FedLearning: Privacy-Preserving Federated Learning for Multi-Center TCM Clinical Data},
  author={ZYY Project},
  year={2025},
  url={https://github.com/YOUR_USERNAME/TCM-FedLearning}
}
```

### Key References

- McMahan, B. et al. Communication-efficient learning of deep networks from decentralized data. *AISTATS* (2017).
- Li, T. et al. Federated optimization in heterogeneous networks. *MLSys* (2020).
- Karimireddy, S.P. et al. SCAFFOLD: Stochastic controlled averaging for federated learning. *ICML* (2020).
- Dwork, C. & Roth, A. The algorithmic foundations of differential privacy. *Foundations and Trends in TCS* (2014).

---

## Roadmap

- [ ] Support for real multi-hospital deployment with gRPC communication
- [ ] Secure aggregation protocol implementation
- [ ] Vertical federated learning for cross-institutional feature spaces
- [ ] Asynchronous federated training for stragglers
- [ ] Integration with TCM-EMR-Temporal for federated syndrome prediction
- [ ] Docker containerization for reproducible deployment
- [ ] Web dashboard for training monitoring and visualization

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, collaborations, or issues, please open a GitHub Issue or contact the project maintainers.
