from .dqn_config import basic_dqn_config
from .random_agent_config import random_agent_config
from .rnd_config import rnd_config
from .cql_config import cql_config
from .awac_config import awac_config
from .iql_config import iql_config
from .diayn_sac_config import diayn_sac_config

configs = {
    "dqn": basic_dqn_config,
    "random": random_agent_config,
    "rnd": rnd_config,
    "cql": cql_config,
    "awac": awac_config,
    "iql": iql_config,
    "diayn_sac": diayn_sac_config,
}
