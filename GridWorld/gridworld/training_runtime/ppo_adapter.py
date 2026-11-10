"""学習画面用のエージェント名。AI内部の状態には触れない。"""

from agents.grag import GRAG
from agents.ppo_agent import PPOAgent


class GRAGSession(GRAG):
    display_name = "GRAG"


class PPOAgentSession(PPOAgent):
    display_name = "PPOエージェント"
