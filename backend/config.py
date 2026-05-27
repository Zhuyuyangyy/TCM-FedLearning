from dataclasses import dataclass

@dataclass
class Settings:
    app_name: str = "TCM-FedLearning"
    version: str = "0.1.0"
    port: int = 8027

settings = Settings()
