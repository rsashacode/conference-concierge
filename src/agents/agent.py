import logging
import sys
from openai import OpenAI

from src.state import AgentState


def init_logging(level=logging.INFO, name="ConferenceConcierge"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%d.%m.%Y %H:%M:%S %z"))
    logger.addHandler(handler)


class Agent:
    """
    An abstract class for all agents.
    """
    def __init__(self, name, model, role, system_prompt, tools):
        init_logging(name=name)
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.client = OpenAI()
        
    def log(self, msg):
        logging.getLogger(self.name).info(msg)
        
    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError("Subclasses must implement this method")
