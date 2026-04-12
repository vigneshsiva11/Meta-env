"""
server/app.py
=============
Wires ApiContractEnvironment into the OpenEnv FastAPI server.

IMPORTANT: pass the CLASS (factory), never an instance.
Each WebSocket session gets its own isolated environment object.
"""
from openenv.core.env_server import create_app
import uvicorn

from models import ContractAction, ContractObservation
from server.environment import ApiContractEnvironment

app = create_app(
    env=ApiContractEnvironment,        # class/factory — NOT an instance
    action_cls=ContractAction,
    observation_cls=ContractObservation,
    env_name="api_contract_env",
    max_concurrent_envs=4,             # safe: SUPPORTS_CONCURRENT_SESSIONS=True
)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
