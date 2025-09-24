# src/graph/state.py
from typing import TypedDict, List, Optional
import pandas as pd

class AgentState(TypedDict):
    """
    Define a estrutura do estado compartilhado entre os nós do grafo.
    Esta é a 'memória' do sistema para cada execução.
    """
    user_question: str
    raw_dataframe: pd.DataFrame
    dataframe_profile: Optional[dict]
    routing_decision: str
    generated_code: Optional[str]
    execution_result: Optional[dict]
    synthesis: Optional[str]
    error_message: Optional[str]
    conversation_history: List[str]
