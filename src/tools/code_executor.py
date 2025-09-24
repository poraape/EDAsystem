# src/tools/code_executor.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import io
import base64
import warnings

def execute_python_code(code: str, df: pd.DataFrame) -> dict:
    """
    Executa código Python em um ambiente controlado para análise e visualização.
    
    AVISO DE SEGURANÇA: Esta implementação usa exec() e é INSEGURA para produção.
    Ela serve como um protótipo funcional. Em um ambiente real, substitua isso
    por uma chamada a um sandbox seguro (ex: um contêiner Docker com limites
    de recursos, sem acesso à rede e com bibliotecas pré-instaladas).
    """
    try:
        # Namespace local para a execução do código
        local_namespace = {
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'np': np,
            'stats': stats,
            'KMeans': KMeans,
            'warnings': warnings,
            'df': df.copy() # Passa uma cópia para evitar modificação do original
        }
        
        # Garante que o código não sobrescreva o namespace principal
        exec("warnings.filterwarnings('ignore')", local_namespace)
        exec(code, {}, local_namespace)
        
        # Verifica se uma figura foi gerada e a converte para base64
        fig = plt.gcf()
        image_base64 = None
        if fig.get_axes():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

        return {
            "success": True,
            "image_base64": image_base64,
            "error": None
        }
    except Exception as e:
        plt.close('all') # Garante que figuras com erro sejam fechadas
        return {"success": False, "error": str(e), "image_base64": None}
