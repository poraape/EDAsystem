# src/graph/builder.py
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

from .state import AgentState
from ..tools.code_executor import execute_python_code

# --- Inicialização do LLM (Gemini) ---
# Será usado por múltiplos agentes
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)

# --- Definição dos Nós (Agentes) ---

def data_ingestion_node(state: AgentState) -> dict:
    """
    Nó responsável por analisar o DataFrame inicial e criar um perfil.
    Este nó não precisa de um LLM.
    """
    if state.get("dataframe_profile") is None:
        df = state["raw_dataframe"]
        profile = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict()
        }
        return {"dataframe_profile": profile}
    return {}

def orchestrator_node(state: AgentState) -> dict:
    """
    Nó que decide qual o próximo passo a ser tomado com base na pergunta do usuário.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um orquestrador de tarefas de IA. Sua função é analisar a pergunta do usuário e o contexto da conversa para decidir a próxima ação.
As opções são:
1. 'generate_code': Se a pergunta exigir uma análise quantitativa, um gráfico ou uma manipulação de dados. Exemplos: "qual a correlação?", "mostre um histograma da coluna X", "quantos valores nulos existem?".
2. 'synthesize': Se a pergunta for uma saudação, uma pergunta geral sobre as conclusões ou um pedido de resumo. Exemplos: "olá", "quais os principais insights até agora?", "resuma o que encontramos".
3. 'end': Se a conversa parece ter terminado ou a pergunta não está relacionada à análise de dados.

Responda apenas com a ação escolhida em letras minúsculas (ex: 'generate_code')."""),
        ("human", f"""Contexto da Conversa: {state['conversation_history']}
Pergunta do Usuário: {state['user_question']}
Perfil dos Dados: {state['dataframe_profile']}
Responda com a próxima ação.""")
    ])
    chain = prompt | llm
    decision = chain.invoke({}).content
    return {"routing_decision": decision}

def code_generation_node(state: AgentState) -> dict:
    """
    Nó que gera e executa código Python para responder à pergunta do usuário.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em análise de dados que gera código Python.
Sua tarefa é escrever um script Python para responder à pergunta do usuário usando o DataFrame `df`.
- Use as bibliotecas pandas, matplotlib, seaborn, numpy, scipy.
- O DataFrame está disponível como uma variável chamada `df`.
- Gere apenas o corpo do código, sem a definição da função.
- Se for gerar um gráfico, use `plt.figure()` para criar uma nova figura e adicione títulos e labels claros. O gráfico será salvo automaticamente.
- Não use `print()`. Apenas o gráfico gerado será retornado.
- O código deve ser seguro e não pode acessar arquivos ou a rede."""),
        ("human", f"""Perfil dos Dados: {state['dataframe_profile']}
Pergunta: {state['user_question']}
Gere o código Python para responder a esta pergunta.""")
    ])
    chain = prompt | llm
    generated_code = chain.invoke({}).content.strip('`').strip('python\n')
    
    execution_result = execute_python_code(generated_code, state["raw_dataframe"])
    
    return {"generated_code": generated_code, "execution_result": execution_result}

def insight_synthesis_node(state: AgentState) -> dict:
    """
    Nó que sintetiza uma resposta em linguagem natural para o usuário.
    """
    context = f"""Pergunta do Usuário: {state['user_question']}
Perfil dos Dados: {state['dataframe_profile']}
Código Gerado: {state.get('generated_code')}
Resultado da Execução: {state.get('execution_result')}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um consultor de dados. Sua tarefa é fornecer uma resposta clara e concisa para o usuário com base no contexto fornecido. Se um gráfico foi gerado, explique os insights que ele revela. Se ocorreu um erro, explique-o de forma simples."),
        ("human", context)
    ])
    chain = prompt | llm
    synthesis = chain.invoke({}).content
    return {"synthesis": synthesis}

# --- Construção do Grafo ---

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("code_generator", code_generation_node)
    workflow.add_node("insight_synthesizer", insight_synthesis_node)

    workflow.set_entry_point("data_ingestion")
    workflow.add_edge("data_ingestion", "orchestrator")
    
    workflow.add_conditional_edges(
        "orchestrator",
        lambda state: state["routing_decision"],
        {
            "generate_code": "code_generator",
            "synthesize": "insight_synthesizer",
            "end": END
        }
    )
    
    workflow.add_edge("code_generator", "insight_synthesizer")
    workflow.add_edge("insight_synthesizer", END)

    return workflow.compile()
