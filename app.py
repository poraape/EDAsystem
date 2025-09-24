# app.py
import streamlit as st
import pandas as pd
from src.graph.builder import build_graph
import base64
import os
from dotenv import load_dotenv

# Carrega a GOOGLE_API_KEY do arquivo .env
load_dotenv()

# --- Configuração da Página ---
st.set_page_config(
    page_title="MAMA-X Ωmega | EDA Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilo CSS Customizado ---
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .st-emotion-cache-16txtl3 { padding: 2rem 1rem 10rem; }
    .st-chat-message { background-color: #ffffff; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- Inicialização do Estado da Sessão ---
if 'graph' not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.messages = []
    st.session_state.raw_df = None
    st.session_state.df_profile = None
    st.session_state.session_id = os.urandom(24).hex()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("MAMA-X Ωmega")
    st.markdown("### Sistema de Análise de Dados Autônomo")
    st.divider()
    st.header("1. Carregar Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv", label_visibility="collapsed")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = df
            st.success("Arquivo carregado!")
            with st.expander("Visualizar Amostra"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            st.session_state.raw_df = None
    
    st.divider()
    st.info("Este protótipo foi construído seguindo as diretrizes do MAMA-X Ωmega.")

# --- Área Principal da Aplicação ---
st.header("🤖 Assistente de Análise Exploratória de Dados")
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message and message["image"]:
                st.image(base64.b64decode(message["image"]), caption="Gráfico Gerado pela Análise")

prompt = st.chat_input("Ex: 'Qual a correlação entre as variáveis?'")

if prompt:
    if st.session_state.raw_df is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral para começar.")
    else:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analisando... Os agentes estão colaborando."):
            initial_state = {
                "user_question": prompt,
                "raw_dataframe": st.session_state.raw_df,
                "dataframe_profile": st.session_state.df_profile,
                "conversation_history": [m['content'] for m in st.session_state.messages]
            }
            config = {"configurable": {"session_id": st.session_state.session_id}}
            final_state = st.session_state.graph.invoke(initial_state, config=config)
            
            if final_state.get("dataframe_profile"):
                st.session_state.df_profile = final_state["dataframe_profile"]

            response_content = final_state.get("synthesis", "Desculpe, não consegui processar sua solicitação.")
            if final_state.get("error_message"):
                response_content += f"\n\n**Ocorreu um erro:** `{final_state['error_message']}`"
            
            image_b64 = final_state.get("execution_result", {}).get("image_base64")

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                    if image_b64:
                        st.image(base64.b64decode(image_b64), caption="Gráfico Gerado pela Análise")
            
            st.session_state.messages.append({"role": "assistant", "content": response_content, "image": image_b64})
