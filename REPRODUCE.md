# REPRODUCE.md - TCM-FedLearning

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux / macOS / Windows
- **GPU**: Not required (CPU sufficient for API server)

## Install

```bash
cd TCM-FedLearning
pip install -e .
```

Or install dependencies directly:
```bash
pip install fastapi uvicorn scikit-learn numpy pandas pyyaml pydantic pydantic-settings torch scipy networkx
```

## Smoke Test

```bash
python -c "from backend.main import app; print('Import OK')"
```

```bash
pytest backend/tests/ -v
```

## Run Server

```bash
uvicorn backend.main:app --port 8024 --reload
```

## API Documentation

Access Swagger UI at: http://localhost:8024/docs

## Project Description

面向多中心中医临床数据的隐私保护联邦学习

## Known Issues

- No external real clinical data included; uses synthetic/demo data
- torch is a heavy dependency; consider CPU-only install for API-only usage
- No hardcoded absolute paths detected in core code
