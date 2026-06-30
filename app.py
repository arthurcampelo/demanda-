# app.py
# Previsor de Demanda Semanal
# Aplicativo acadêmico em Python + Streamlit para Administração da Produção.
# Objetivo: receber demandas históricas, aplicar métodos simples de previsão,
# comparar modelos, gerar gráficos, indicadores e recomendações gerenciais.

from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(
    page_title="Previsor de Demanda Semanal",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# ESTILO VISUAL PROFISSIONAL
# =========================================================
st.markdown(
    """
    <style>
        :root {
            --azul: #12355B;
            --azul-logo: #143D73;
            --verde-logo: #31E981;
            --verde: #1B7F5C;
            --laranja: #D97A07;
            --cinza: #F4F7FA;
            --borda: #CBD5E1;
            --texto: #0F172A;
        }
        .stApp { background: #F6F9FC; color: var(--texto); }
        .block-container { padding-top: 1.2rem; }
        .hero {
            background: linear-gradient(135deg, #143D73 0%, #12355B 55%, #1B7F5C 100%);
            padding: 24px 28px;
            border-radius: 6px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 10px 24px rgba(18,53,91,0.20);
            border: 1px solid rgba(255,255,255,0.18);
        }
        .hero h1 { margin: 0; font-size: 2.2rem; font-weight: 900; letter-spacing: -0.03em; }
        .hero p { margin-top: 8px; font-size: 1.02rem; line-height: 1.5; opacity: .96; }
        .hero-badge {
            display: inline-block; background: rgba(49,233,129,0.18);
            border: 1px solid rgba(49,233,129,0.42); color: #FFFFFF;
            padding: 6px 12px; border-radius: 4px; font-weight: 800; margin-bottom: 10px;
        }
        .brand-logo-box {
            background: #FFFFFF; border-radius: 6px; padding: 12px;
            box-shadow: 0 8px 20px rgba(15,23,42,0.12);
            border: 2px solid rgba(49,233,129,0.35);
        }
        .card, div[data-testid="stExpander"] {
            border: 1px solid var(--borda); background: #FFFFFF;
            border-radius: 6px !important; box-shadow: 0 2px 10px rgba(15,23,42,0.05);
        }
        .small { color: #475569; font-size: 0.92rem; }
        div[data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid var(--borda); border-left: 6px solid var(--azul);
            padding: 14px; border-radius: 6px; box-shadow: 0 2px 10px rgba(15,23,42,0.05);
        }
        div[data-testid="stMetricValue"] { color: var(--azul); font-weight: 800; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
        .stTabs [data-baseweb="tab"] {
            background-color: #EAF0F6; border-radius: 4px 4px 0 0; padding: 10px 14px;
            color: var(--texto) !important; font-weight: 750; border: 1px solid #D8E0E8;
        }
        .stTabs [aria-selected="true"] { background: #12355B !important; color: white !important; }
        .stButton button, .stDownloadButton button {
            border-radius: 4px !important; font-weight: 800 !important; border: 1px solid #12355B !important;
            background: #12355B !important; color: #FFFFFF !important;
        }
        .stButton button:hover, .stDownloadButton button:hover { background: #1B7F5C !important; color: #FFFFFF !important; }
        .alerta { border-left: 6px solid #D97A07; background: #FFF7ED; padding: 12px 14px; border-radius: 4px; margin: 10px 0; color:#111827; }
        .ok { border-left: 6px solid #1B7F5C; background: #ECFDF5; padding: 12px 14px; border-radius: 4px; margin: 10px 0; color:#111827; }

        /* Sidebar clara e legível: evita textos transparentes nos controles */
        section[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #CBD5E1; }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
            color: #0F172A !important;
        }
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stNumberInput label,
        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stTextArea label,
        section[data-testid="stSidebar"] .stFileUploader label {
            color: #0F172A !important; font-weight: 800 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            color: #0F172A !important; background-color: #FFFFFF !important;
            -webkit-text-fill-color: #0F172A !important;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label div,
        section[data-testid="stSidebar"] [role="radiogroup"] label p { color: #0F172A !important; }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: #F8FAFC !important; border: 1px solid #CBD5E1 !important; border-radius: 4px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary * { color: #0F172A !important; font-weight: 800 !important; }
        section[data-testid="stSidebar"] .stDownloadButton button, section[data-testid="stSidebar"] .stButton button {
            width: 100%; color: #FFFFFF !important; background: #143D73 !important; border-radius: 4px !important;
        }
        .sidebar-brand {
            background: #FFFFFF;
            color: #0F172A !important;
            padding: 16px 14px;
            border-radius: 14px;
            margin-bottom: 16px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
            text-align: center;
        }
        .sidebar-brand h2 {
            margin: 0 !important;
            color: #143D73 !important;
            font-weight: 900 !important;
            letter-spacing: -0.5px;
        }
        .sidebar-brand p {
            margin: 4px 0 0 0 !important;
            color: #64748B !important;
            font-size: 0.86rem;
        }
        .sidebar-section-title {
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #143D73 !important;
            margin: 18px 0 8px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid #E2E8F0;
        }
        .sidebar-note {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            color: #334155 !important;
            padding: 10px 12px;
            border-radius: 12px;
            font-size: 0.86rem;
            line-height: 1.35;
            margin-bottom: 10px;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stRadio > label,
        section[data-testid="stSidebar"] .stSelectbox > label,
        section[data-testid="stSidebar"] .stSlider > label,
        section[data-testid="stSidebar"] .stNumberInput > label,
        section[data-testid="stSidebar"] .stTextInput > label,
        section[data-testid="stSidebar"] .stTextArea > label,
        section[data-testid="stSidebar"] .stFileUploader > label {
            color: #143D73 !important;
            font-weight: 800 !important;
        }
        @media (max-width: 768px) {
            .hero { padding: 18px 16px; border-radius: 4px; }
            .hero h1 { font-size: 1.65rem; }
            .hero p { font-size: 0.92rem; }
            .brand-logo-box { max-width: 180px; margin: 0 auto 10px auto; }
            div[data-testid="stMetric"] { padding: 10px; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FUNÇÕES DE PREPARAÇÃO DOS DADOS
# =========================================================
def parse_lista_demandas(texto: str) -> List[float]:
    """Converte o texto digitado pelo usuário em lista de demandas numéricas."""
    if not texto or not texto.strip():
        return []
    texto = texto.replace(";", ",").replace("\n", ",")
    valores: List[float] = []
    for parte in texto.split(","):
        parte = parte.strip().replace(" ", "")
        if parte:
            # Aceita 123,45 e 123.45 quando o separador da lista já foi tratado.
            parte = parte.replace(".", ".").replace(",", ".")
            valores.append(float(parte))
    return valores


def carregar_dados_upload(arquivo) -> pd.DataFrame:
    """Lê CSV ou Excel e padroniza colunas principais quando possível."""
    nome = arquivo.name.lower()
    if nome.endswith(".csv"):
        df = pd.read_csv(arquivo)
    elif nome.endswith((".xlsx", ".xls")):
        df = pd.read_excel(arquivo)
    else:
        raise ValueError("Formato não aceito. Use CSV ou Excel.")

    df.columns = [str(c).strip() for c in df.columns]
    mapa = {c.lower(): c for c in df.columns}

    # Padronização flexível de nomes.
    candidatos = {
        "Semana": ["semana", "week"],
        "Demanda": ["demanda", "vendas", "pedidos", "quantidade", "demand"],
        "Produto": ["produto", "product"],
        "Regiao": ["regiao", "região", "regional", "cidade", "local", "region"],
        "Promocao": ["promocao", "promoção", "promo", "promotion"],
        "Feriado": ["feriado", "holiday"],
        "Evento": ["evento", "event"],
        "Preco": ["preco", "preço", "price"],
    }
    renomear = {}
    for padrao, opcoes in candidatos.items():
        for opc in opcoes:
            if opc in mapa:
                renomear[mapa[opc]] = padrao
                break
    df = df.rename(columns=renomear)

    if "Demanda" not in df.columns:
        raise ValueError("A planilha precisa ter uma coluna de demanda, vendas, pedidos ou quantidade.")
    if "Semana" not in df.columns:
        df.insert(0, "Semana", range(1, len(df) + 1))
    if "Produto" not in df.columns:
        df["Produto"] = "Produto A"
    if "Regiao" not in df.columns:
        df["Regiao"] = "Geral"

    df["Semana"] = pd.to_numeric(df["Semana"], errors="coerce")
    df["Demanda"] = pd.to_numeric(df["Demanda"], errors="coerce")
    df = df.dropna(subset=["Semana", "Demanda"]).copy()
    df["Semana"] = df["Semana"].astype(int)
    df["Demanda"] = df["Demanda"].astype(float)
    return df.sort_values(["Produto", "Regiao", "Semana"]).reset_index(drop=True)


def gerar_df_manual(produto: str, demandas: List[float], regiao: str = "Geral") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Semana": list(range(1, len(demandas) + 1)),
            "Produto": produto or "Produto A",
            "Regiao": regiao or "Geral",
            "Demanda": demandas,
        }
    )




def gerar_comercio_simulado() -> pd.DataFrame:
    """Cria uma base realista de um pequeno comércio para demonstração e testes.

    A base simula uma mercearia/padaria de bairro com produtos de giro semanal,
    regiões de venda, promoções, feriados/eventos, estoque e capacidade.
    Os valores são fictícios, mas seguem lógica gerencial realista: produtos básicos
    têm demanda mais estável; itens de fim de semana e promoção têm picos.
    """
    produtos = [
        {"Produto": "Pão francês", "Categoria": "Padaria", "base": 820, "Preco": 0.75, "CustoUnitario": 0.42, "EstoqueAtual": 180, "EstoqueMinimo": 120, "CapacidadeSemanal": 1050},
        {"Produto": "Bolo caseiro", "Categoria": "Padaria", "base": 95, "Preco": 18.00, "CustoUnitario": 9.50, "EstoqueAtual": 22, "EstoqueMinimo": 15, "CapacidadeSemanal": 140},
        {"Produto": "Cuscuz pronto", "Categoria": "Lanches", "base": 180, "Preco": 6.00, "CustoUnitario": 3.10, "EstoqueAtual": 45, "EstoqueMinimo": 30, "CapacidadeSemanal": 260},
        {"Produto": "Marmita executiva", "Categoria": "Almoço", "base": 260, "Preco": 16.00, "CustoUnitario": 9.20, "EstoqueAtual": 55, "EstoqueMinimo": 45, "CapacidadeSemanal": 360},
        {"Produto": "Suco natural", "Categoria": "Bebidas", "base": 210, "Preco": 7.00, "CustoUnitario": 3.30, "EstoqueAtual": 65, "EstoqueMinimo": 40, "CapacidadeSemanal": 320},
        {"Produto": "Café 250g", "Categoria": "Mercearia", "base": 135, "Preco": 12.50, "CustoUnitario": 8.20, "EstoqueAtual": 38, "EstoqueMinimo": 28, "CapacidadeSemanal": 210},
        {"Produto": "Arroz 1kg", "Categoria": "Mercearia", "base": 160, "Preco": 7.20, "CustoUnitario": 5.40, "EstoqueAtual": 48, "EstoqueMinimo": 35, "CapacidadeSemanal": 240},
        {"Produto": "Água mineral", "Categoria": "Bebidas", "base": 300, "Preco": 3.00, "CustoUnitario": 1.45, "EstoqueAtual": 90, "EstoqueMinimo": 60, "CapacidadeSemanal": 450},
    ]
    regioes = ["Centro", "Bairro", "Delivery"]
    pesos_regiao = {"Centro": 0.48, "Bairro": 0.34, "Delivery": 0.18}
    eventos = {4: "Promoção de inauguração", 8: "Feriado local", 12: "Evento escolar", 18: "Promoção relâmpago", 22: "Semana de pagamento", 26: "Festa junina"}
    linhas = []
    for prod_idx, prod in enumerate(produtos):
        for semana in range(1, 27):
            tendencia = 1 + (semana - 1) * (0.006 if prod["Categoria"] in ["Bebidas", "Almoço"] else 0.003)
            sazonal = 1 + (0.10 if semana % 4 == 0 else 0) - (0.05 if semana % 5 == 0 else 0)
            promocao = 1 if semana in [4, 18] and prod["Categoria"] in ["Padaria", "Bebidas"] else 0
            feriado = 1 if semana in [8, 26] else 0
            fator_evento = 1 + (0.18 * promocao) + (0.12 * feriado) + (0.10 if semana == 12 and prod["Categoria"] in ["Lanches", "Bebidas"] else 0)
            # ruído determinístico para evitar dependência de aleatoriedade e facilitar testes
            ruido = 1 + (((semana * 7 + prod_idx * 5) % 9) - 4) / 100
            demanda_total = prod["base"] * tendencia * sazonal * fator_evento * ruido
            for regiao in regioes:
                linhas.append({
                    "Semana": semana,
                    "Produto": prod["Produto"],
                    "Categoria": prod["Categoria"],
                    "Regiao": regiao,
                    "Demanda": round(demanda_total * pesos_regiao[regiao]),
                    "Preco": prod["Preco"],
                    "CustoUnitario": prod["CustoUnitario"],
                    "EstoqueAtual": prod["EstoqueAtual"],
                    "EstoqueMinimo": prod["EstoqueMinimo"],
                    "CapacidadeSemanal": prod["CapacidadeSemanal"],
                    "Promocao": promocao,
                    "Feriado": feriado,
                    "Evento": eventos.get(semana, "Rotina"),
                })
    return pd.DataFrame(linhas)


def agregar_demanda(df: pd.DataFrame, produto: str, regiao: str) -> pd.DataFrame:
    base = df.copy()
    if produto != "Todos":
        base = base[base["Produto"] == produto]
    if regiao != "Todas":
        base = base[base["Regiao"] == regiao]
    base = base.groupby("Semana", as_index=False)["Demanda"].sum().sort_values("Semana")
    return base.reset_index(drop=True)


# =========================================================
# MÉTODOS DE PREVISÃO
# =========================================================
def metodo_ingenuo(y: List[float], h: int) -> List[float]:
    return [max(0.0, float(y[-1]))] * h


def media_movel(y: List[float], h: int, janela: int = 3) -> List[float]:
    hist = list(map(float, y))
    out = []
    for _ in range(h):
        j = min(janela, len(hist))
        p = float(np.mean(hist[-j:]))
        out.append(max(0.0, p))
        hist.append(p)
    return out


def media_ponderada(y: List[float], h: int, pesos: Optional[List[float]] = None) -> List[float]:
    hist = list(map(float, y))
    if pesos is None:
        pesos_np = np.array([0.2, 0.3, 0.5], dtype=float)
    else:
        pesos_np = np.array(pesos, dtype=float)
    pesos_np = pesos_np / pesos_np.sum()
    janela = len(pesos_np)
    out = []
    for _ in range(h):
        j = min(janela, len(hist))
        serie = np.array(hist[-j:], dtype=float)
        pesos_j = pesos_np[-j:]
        pesos_j = pesos_j / pesos_j.sum()
        p = float(np.sum(serie * pesos_j))
        out.append(max(0.0, p))
        hist.append(p)
    return out


def suavizacao_exponencial(y: List[float], h: int, alfa: float = 0.30) -> List[float]:
    serie = list(map(float, y))
    f = serie[0]
    for real in serie[1:]:
        f = alfa * real + (1 - alfa) * f
    return [max(0.0, float(f))] * h


def regressao_linear(y: List[float], h: int) -> List[float]:
    serie = np.array(y, dtype=float)
    x = np.arange(1, len(serie) + 1).reshape(-1, 1)
    modelo = LinearRegression().fit(x, serie)
    futuras = np.arange(len(serie) + 1, len(serie) + h + 1).reshape(-1, 1)
    pred = modelo.predict(futuras)
    return [max(0.0, float(v)) for v in pred]


def prever(metodo: str, y: List[float], h: int, janela: int, alfa: float) -> List[float]:
    if metodo == "Ingênuo":
        return metodo_ingenuo(y, h)
    if metodo == "Média móvel simples":
        return media_movel(y, h, janela)
    if metodo == "Média móvel ponderada":
        return media_ponderada(y, h)
    if metodo == "Suavização exponencial simples":
        return suavizacao_exponencial(y, h, alfa)
    if metodo == "Regressão linear simples":
        return regressao_linear(y, h)
    return media_movel(y, h, janela)


def backtest(metodo: str, y: List[float], janela: int, alfa: float) -> pd.DataFrame:
    """Faz previsão um passo à frente usando apenas o histórico anterior."""
    minimo = 3 if metodo != "Regressão linear simples" else 4
    linhas = []
    if len(y) <= minimo:
        return pd.DataFrame(columns=["Semana", "Real", "Previsto", "Erro", "Erro absoluto", "Erro percentual"])
    for i in range(minimo, len(y)):
        hist = y[:i]
        real = float(y[i])
        pred = float(prever(metodo, hist, 1, janela, alfa)[0])
        erro = real - pred
        erro_pct = abs(erro) / real * 100 if real != 0 else np.nan
        linhas.append(
            {
                "Semana": i + 1,
                "Real": real,
                "Previsto": pred,
                "Erro": erro,
                "Erro absoluto": abs(erro),
                "Erro percentual": erro_pct,
            }
        )
    return pd.DataFrame(linhas)


def metricas_erro(bt: pd.DataFrame) -> Dict[str, Optional[float]]:
    if bt.empty:
        return {"MAE": None, "RMSE": None, "MAPE": None, "Precisão": None}
    mae = float(bt["Erro absoluto"].mean())
    rmse = float(math.sqrt(mean_squared_error(bt["Real"], bt["Previsto"])))
    mape = float(bt["Erro percentual"].dropna().mean()) if bt["Erro percentual"].notna().any() else None
    precisao = max(0.0, 100 - mape) if mape is not None else None
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Precisão": precisao}


def comparar_modelos(y: List[float], h: int, janela: int, alfa: float) -> pd.DataFrame:
    metodos = [
        "Ingênuo",
        "Média móvel simples",
        "Média móvel ponderada",
        "Suavização exponencial simples",
        "Regressão linear simples",
    ]
    linhas = []
    for m in metodos:
        pred = prever(m, y, h, janela, alfa)
        bt = backtest(m, y, janela, alfa)
        met = metricas_erro(bt)
        linhas.append(
            {
                "Modelo": m,
                "Previsão próxima semana": round(pred[0], 2),
                "MAE": None if met["MAE"] is None else round(met["MAE"], 2),
                "RMSE": None if met["RMSE"] is None else round(met["RMSE"], 2),
                "MAPE (%)": None if met["MAPE"] is None else round(met["MAPE"], 2),
                "Precisão estimada (%)": None if met["Precisão"] is None else round(met["Precisão"], 2),
            }
        )
    return pd.DataFrame(linhas)


# =========================================================
# ANÁLISES GERENCIAIS
# =========================================================
def analisar_tendencia(y: List[float]) -> Tuple[str, float, float]:
    serie = np.array(y, dtype=float)
    if len(serie) < 4:
        return "dados insuficientes", 0.0, 0.0
    media = float(np.mean(serie))
    if media == 0:
        return "estável", 0.0, 0.0
    x = np.arange(1, len(serie) + 1).reshape(-1, 1)
    modelo = LinearRegression().fit(x, serie)
    inclinacao = float(modelo.coef_[0])
    cv = float(np.std(serie) / media)
    limite = max(media * 0.02, 1.0)
    if cv > 0.25:
        return "irregular", inclinacao, cv
    if inclinacao > limite:
        return "crescente", inclinacao, cv
    if inclinacao < -limite:
        return "decrescente", inclinacao, cv
    return "estável", inclinacao, cv


def intervalo_confianca(previsoes: List[float], erros_abs: pd.Series, z: float = 1.96) -> pd.DataFrame:
    if erros_abs is None or len(erros_abs.dropna()) == 0:
        margem = 0.10 * max(1, float(np.mean(previsoes)))
    else:
        desvio = float(np.std(erros_abs.dropna(), ddof=1)) if len(erros_abs.dropna()) > 1 else float(erros_abs.mean())
        margem = max(desvio * z, 0.05 * max(1, float(np.mean(previsoes))))
    return pd.DataFrame(
        {
            "Previsão": previsoes,
            "Limite inferior": [max(0.0, p - margem) for p in previsoes],
            "Limite superior": [p + margem for p in previsoes],
        }
    )


def detectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    if len(base) < 4:
        base["Anomalia"] = False
        base["Z-score"] = 0.0
        return base
    media = base["Demanda"].mean()
    desvio = base["Demanda"].std(ddof=0)
    if desvio == 0 or pd.isna(desvio):
        base["Z-score"] = 0.0
        base["Anomalia"] = False
    else:
        base["Z-score"] = (base["Demanda"] - media) / desvio
        base["Anomalia"] = base["Z-score"].abs() >= 2.0
    return base


def analisar_sazonalidade(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["Ciclo de 4 semanas"] = ((base["Semana"] - 1) % 4) + 1
    saz = base.groupby("Ciclo de 4 semanas", as_index=False)["Demanda"].mean()
    saz = saz.rename(columns={"Demanda": "Demanda média"})
    return saz


def feature_importance(df_original: pd.DataFrame) -> Optional[pd.DataFrame]:
    variaveis = [c for c in ["Promocao", "Feriado", "Evento", "Preco"] if c in df_original.columns]
    if len(variaveis) == 0 or len(df_original) < 8:
        return None
    base = df_original.dropna(subset=["Demanda"]).copy()
    X = pd.DataFrame()
    for c in variaveis:
        if base[c].dtype == object:
            X[c] = pd.factorize(base[c].astype(str))[0]
        else:
            X[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)
    y = base["Demanda"].astype(float)
    if X.shape[0] < 8:
        return None
    modelo = RandomForestRegressor(n_estimators=120, random_state=42)
    modelo.fit(X, y)
    imp = pd.DataFrame({"Variável": X.columns, "Importância": modelo.feature_importances_})
    return imp.sort_values("Importância", ascending=False).reset_index(drop=True)


def gerar_recomendacao(
    tendencia: str,
    previsoes: List[float],
    media_hist: float,
    capacidade: float,
    estoque_atual: float,
    estoque_seg: float,
    anomalias: int,
    melhor_modelo: str,
) -> str:
    media_prev = float(np.mean(previsoes))
    variacao = ((media_prev - media_hist) / media_hist * 100) if media_hist else 0
    partes = []

    if tendencia == "crescente":
        partes.append("A demanda apresenta tendência de crescimento. Recomenda-se avaliar aumento de produção, compras de matéria-prima e capacidade da equipe.")
    elif tendencia == "decrescente":
        partes.append("A demanda apresenta tendência de queda. Recomenda-se reduzir o risco de excesso de estoque e evitar produzir acima da necessidade.")
    elif tendencia == "irregular":
        partes.append("A demanda está irregular. Recomenda-se investigar promoções, eventos, feriados, clima, rupturas de abastecimento ou mudanças no comportamento do cliente.")
    elif tendencia == "estável":
        partes.append("A demanda está relativamente estável. A previsão pode ser usada como referência para manter um ritmo produtivo próximo ao histórico.")
    else:
        partes.append("Há poucos dados para uma conclusão segura. É recomendável registrar mais semanas antes de decisões produtivas importantes.")

    if variacao > 10:
        partes.append("A média prevista está acima da média histórica, indicando risco de falta de produto caso a produção não acompanhe o crescimento.")
    elif variacao < -10:
        partes.append("A média prevista está abaixo da média histórica, indicando risco de formação de estoque se a produção permanecer no mesmo nível anterior.")
    else:
        partes.append("A média prevista está próxima da média histórica, sugerindo ajustes moderados no planejamento.")

    if capacidade > 0:
        maior = max(previsoes)
        if maior > capacidade:
            partes.append(f"Alerta de capacidade: a maior previsão ({maior:.0f}) supera a capacidade semanal informada ({capacidade:.0f}).")
        else:
            partes.append("A capacidade informada aparenta ser suficiente para cobrir a previsão, mantendo acompanhamento semanal.")

    necessidade = max(0.0, sum(previsoes) + estoque_seg - estoque_atual)
    partes.append(f"Sugestão de reposição/produção para o horizonte escolhido: aproximadamente {necessidade:.0f} unidades, considerando estoque atual e estoque de segurança informados.")

    if anomalias > 0:
        partes.append(f"Foram detectadas {anomalias} possível(is) anomalia(s). Antes de confiar na previsão, verifique se houve evento incomum nessas semanas.")

    if melhor_modelo:
        partes.append(f"Na comparação histórica, o modelo com menor erro foi {melhor_modelo}. Isso não garante melhor desempenho futuro, mas ajuda na escolha inicial.")

    partes.append("A previsão deve apoiar a decisão gerencial, mas não deve ser tratada como certeza.")
    return " ".join(partes)


def criar_relatorio_texto(contexto: Dict[str, str], tabelas: Dict[str, pd.DataFrame], recomendacao: str) -> str:
    linhas = []
    linhas.append("RELATÓRIO DO PREVISOR DE DEMANDA SEMANAL")
    linhas.append(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    linhas.append("")
    for k, v in contexto.items():
        linhas.append(f"{k}: {v}")
    linhas.append("")
    linhas.append("RECOMENDAÇÃO GERENCIAL")
    linhas.append(recomendacao)
    linhas.append("")
    for nome, tabela in tabelas.items():
        linhas.append(nome.upper())
        linhas.append(tabela.to_string(index=False))
        linhas.append("")
    return "\n".join(linhas)


# =========================================================
# ESTADO DA SESSÃO
# =========================================================
if "historico_previsoes" not in st.session_state:
    st.session_state["historico_previsoes"] = []
if "historico_alteracoes" not in st.session_state:
    st.session_state["historico_alteracoes"] = []



# =========================================================
# CONTEÚDOS DO ESTUDO GERENCIAL INTEGRADOS AO APP
# =========================================================
def tabela_requisitos_app() -> pd.DataFrame:
    return pd.DataFrame([
        ["RF01", "Cadastro de produtos", "Registrar produto, categoria, preço, custo, estoque e capacidade.", "Funcional", "Obrigatória"],
        ["RF02", "Entrada de demanda histórica", "Receber demanda semanal por base cadastrada, digitação manual ou planilha.", "Funcional", "Obrigatória"],
        ["RF03", "Previsão multissemanal", "Projetar a demanda das próximas semanas conforme horizonte escolhido.", "Funcional", "Obrigatória"],
        ["RF04", "Métodos de previsão", "Aplicar média móvel, média ponderada, suavização exponencial, regressão linear e método ingênuo.", "Funcional", "Obrigatória"],
        ["RF05", "Comparação de modelos", "Comparar erro histórico e indicar o modelo com menor MAE.", "Funcional", "Desejável"],
        ["RF06", "Gráficos e indicadores", "Exibir histórico, previsão, KPIs, tendência, erros e intervalos aproximados.", "Funcional", "Obrigatória"],
        ["RF07", "Recomendação gerencial", "Gerar orientações sobre produção, estoque, reposição e capacidade.", "Funcional", "Obrigatória"],
        ["RF08", "Exportação", "Permitir baixar relatórios em TXT, CSV e Excel.", "Funcional", "Desejável"],
        ["RNF01", "Usabilidade", "Interface clara, em português, com linguagem acessível ao gestor.", "Não funcional", "Obrigatória"],
        ["RNF02", "Confiabilidade", "Bloquear dados inválidos e alertar quando houver poucos dados.", "Não funcional", "Obrigatória"],
        ["RNF03", "Responsividade", "Layout utilizável em notebook e celular.", "Não funcional", "Desejável"],
        ["RNF04", "Transparência", "Informar que previsão apoia a decisão, mas não representa certeza absoluta.", "Não funcional", "Obrigatória"],
    ], columns=["Código", "Requisito", "Descrição", "Tipo", "Prioridade"])


def tabela_variaveis_app() -> pd.DataFrame:
    return pd.DataFrame([
        ["Produto", "Texto", "Identifica qual item será analisado e permite comparar linhas de venda."],
        ["Categoria", "Texto", "Agrupa produtos semelhantes para análise comercial e produtiva."],
        ["Semana", "Número/Data", "Organiza a série histórica e permite cálculo temporal."],
        ["Demanda real", "Número", "Base principal para prever vendas/pedidos futuros."],
        ["Produção realizada", "Número", "Permite comparar produção planejada com demanda efetiva."],
        ["Estoque atual", "Número", "Ajuda a avaliar risco de falta ou excesso de produto."],
        ["Estoque mínimo", "Número", "Serve como estoque de segurança para sugerir reposição."],
        ["Capacidade semanal", "Número", "Mostra o limite produtivo e alerta sobre sobrecarga."],
        ["Preço de venda", "Decimal", "Permite estimar receita prevista e impacto financeiro."],
        ["Custo unitário", "Decimal", "Permite estimar margem e consequência do desperdício."],
        ["Promoção", "Sim/Não", "Pode elevar a demanda temporariamente."],
        ["Feriado ou evento", "Sim/Não/Texto", "Pode alterar o comportamento de compra e gerar sazonalidade."],
        ["Região", "Texto", "Mostra diferenças de demanda por localidade ou canal de venda."],
        ["Falhas ou desperdício", "Número/Texto", "Ajuda a investigar perdas, ruptura e problemas operacionais."],
    ], columns=["Variável", "Tipo de dado", "Justificativa"])


def tabela_testes_app() -> pd.DataFrame:
    return pd.DataFrame([
        ["Demanda estável", "100,100,100,100", "Previsão próxima de 100 e tendência estável."],
        ["Crescimento", "100,110,120,130", "Previsão crescente e alerta para revisar capacidade."],
        ["Queda", "200,180,160,140", "Previsão decrescente e recomendação de cautela no estoque."],
        ["Média móvel", "155,160,165", "Média móvel de 3 semanas próxima de 160."],
        ["Poucos dados", "100,120", "Sistema deve alertar baixa confiabilidade."],
        ["Valor negativo", "-50", "Sistema deve bloquear ou alertar dado inválido."],
        ["Comparação de modelos", "Série variada", "Tabela deve indicar menor erro histórico."],
        ["Gráficos", "Dados válidos", "Histórico e previsão devem aparecer no painel."],
        ["Exportação", "Clique nos botões", "Relatórios TXT, CSV e Excel devem baixar corretamente."],
        ["Responsividade", "Abrir no celular", "Layout deve continuar legível e utilizável."],
    ], columns=["Teste", "Entrada", "Resultado esperado"])


def resumo_estudo_markdown() -> str:
    return """
**Problema de produção:** a empresa precisa planejar quanto produzir nas próximas semanas, mas a demanda varia conforme histórico, promoções, eventos, região e comportamento dos clientes. Prever mal pode gerar falta de produto, excesso de estoque, desperdício, compra incorreta de matéria-prima, sobrecarga da equipe e perda de vendas.

**Solução digital:** o Demanda+ transforma dados históricos em previsões semanais, compara modelos, apresenta gráficos, estima erros e converte os números em recomendações gerenciais para produção, estoque e reposição.

**Modelo mais indicado:** a suavização exponencial simples é destacada por reagir melhor aos dados recentes. A regressão linear ajuda quando há tendência clara, enquanto a média móvel funciona como referência simples e conservadora. O aplicativo compara todos os métodos porque nenhum modelo é perfeito em todas as situações.

**Decisão gerencial:** a previsão não é certeza. O gestor deve avaliar contexto, sazonalidade, promoções, capacidade produtiva, estoque de segurança, qualidade dos dados e comportamento do mercado antes de decidir.
"""


def recomendacoes_praticas_markdown() -> str:
    return """
1. **Ajustar gradualmente a produção:** quando a média prevista fica acima da média histórica, a empresa deve aumentar a produção com cautela para evitar ruptura de estoque sem criar excesso.
2. **Reforçar o estoque de segurança:** quando há promoções, eventos, anomalias ou alta variação, o estoque mínimo reduz risco de falta de produto e perda de vendas.
3. **Monitorar semanalmente o erro da previsão:** comparar demanda prevista e real permite corrigir o planejamento, revisar o método utilizado e melhorar a qualidade da decisão.
"""

# =========================================================
# CABEÇALHO E IDENTIDADE VISUAL
# =========================================================
BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "assets" / "logo_demanda_plus.png"

col_logo, col_hero = st.columns([1, 4], vertical_alignment="center")
with col_logo:
    if LOGO_PATH.exists():
        st.markdown('<div class="brand-logo-box">', unsafe_allow_html=True)
        st.image(str(LOGO_PATH), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
with col_hero:
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">Demanda+ | Previsão com apoio gerencial</span>
            <h1>Previsor de Demanda Semanal</h1>
            <p>Sistema de apoio ao planejamento da produção com previsão de demanda, comparação de modelos, indicadores e recomendações gerenciais.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Explicação da previsão e objetivo do aplicativo", expanded=True):
    st.write(
        "O aplicativo recebe dados históricos de demanda semanal, aplica métodos simples de previsão "
        "e transforma os resultados em apoio à decisão produtiva. Ele ajuda a identificar risco de falta "
        "de produto, excesso de estoque, tendência de crescimento ou queda e necessidade de revisão da capacidade produtiva."
    )
    st.info(
        "A previsão de demanda não é certeza. Ela deve ser analisada junto com contexto, sazonalidade, promoções, eventos, capacidade e qualidade dos dados."
    )


# =========================================================
# SIDEBAR: ENTRADAS E FILTROS
# =========================================================
st.sidebar.markdown("""<div class="sidebar-brand"><h2>Demanda+</h2><p>Painel de previsão semanal</p></div>""", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-note">Escolha a fonte dos dados, filtre o produto e ajuste a previsão. As opções avançadas ficam recolhidas para manter o painel limpo.</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-section-title">Dados</div>', unsafe_allow_html=True)
modo = st.sidebar.radio(
    "Fonte dos dados",
    ["Comércio cadastrado", "Digitação manual", "Upload CSV/Excel"],
)

if modo == "Comércio cadastrado":
    if "df_comercio" not in st.session_state:
        st.session_state["df_comercio"] = gerar_comercio_simulado()
    df_base = st.session_state["df_comercio"].copy()

    with st.sidebar.expander("Cadastrar novo produto", expanded=False):
        novo_produto = st.text_input("Nome do produto", "Tapioca recheada")
        nova_categoria = st.text_input("Categoria", "Lanches")
        nova_regiao = st.selectbox("Região inicial", ["Centro", "Bairro", "Delivery", "Geral"])
        demanda_base = st.number_input("Demanda semanal inicial", min_value=0, value=90, step=5)
        preco_novo = st.number_input("Preço de venda", min_value=0.0, value=8.0, step=0.5)
        custo_novo = st.number_input("Custo unitário", min_value=0.0, value=4.5, step=0.5)
        estoque_novo = st.number_input("Estoque atual do produto", min_value=0, value=30, step=5)
        minimo_novo = st.number_input("Estoque mínimo", min_value=0, value=20, step=5)
        capacidade_nova = st.number_input("Capacidade semanal", min_value=0, value=150, step=10)
        if st.button("Cadastrar produto"):
            linhas = []
            for semana in range(1, 13):
                fator = 1 + (semana - 1) * 0.01 + (0.08 if semana % 4 == 0 else 0)
                linhas.append({
                    "Semana": semana, "Produto": novo_produto, "Categoria": nova_categoria,
                    "Regiao": nova_regiao, "Demanda": round(demanda_base * fator),
                    "Preco": preco_novo, "CustoUnitario": custo_novo,
                    "EstoqueAtual": estoque_novo, "EstoqueMinimo": minimo_novo,
                    "CapacidadeSemanal": capacidade_nova, "Promocao": 0, "Feriado": 0, "Evento": "Cadastro manual",
                })
            st.session_state["df_comercio"] = pd.concat([df_base, pd.DataFrame(linhas)], ignore_index=True)
            st.success("Produto cadastrado na base da sessão.")
            st.rerun()

    st.sidebar.download_button(
        "Baixar base em CSV",
        data=st.session_state["df_comercio"].to_csv(index=False).encode("utf-8-sig"),
        file_name="base_comercio_demanda_plus.csv",
        mime="text/csv",
    )
    df_original = st.session_state["df_comercio"].copy()

elif modo == "Digitação manual":
    produto_manual = st.sidebar.text_input("Produto", "Produto A")
    regiao_manual = st.sidebar.text_input("Região", "Geral")
    texto = st.sidebar.text_area(
        "Demandas históricas semanais",
        "120, 125, 130, 128, 135, 140, 145, 150, 148, 155, 160, 165",
        help="Digite valores separados por vírgula. O roteiro recomenda pelo menos 8 a 12 semanas.",
    )
    try:
        demandas_manual = parse_lista_demandas(texto)
        df_original = gerar_df_manual(produto_manual, demandas_manual, regiao_manual)
    except Exception:
        st.error("Erro ao ler as demandas. Use apenas números separados por vírgula, como: 100, 110, 120.")
        st.stop()
else:
    arquivo = st.sidebar.file_uploader("Enviar arquivo CSV ou Excel", type=["csv", "xlsx", "xls"])
    if arquivo is None:
        st.warning("Envie uma planilha ou use a digitação manual para iniciar.")
        st.stop()
    try:
        df_original = carregar_dados_upload(arquivo)
    except Exception as e:
        st.error(str(e))
        st.stop()

if df_original.empty:
    st.warning("Não há dados válidos para calcular a previsão.")
    st.stop()
if (df_original["Demanda"] < 0).any():
    st.error("A demanda não pode ser negativa. Corrija os valores negativos antes de continuar.")
    st.stop()

produtos = ["Todos"] + sorted(df_original["Produto"].astype(str).unique().tolist())
regioes = ["Todas"] + sorted(df_original["Regiao"].astype(str).unique().tolist())

st.sidebar.markdown('<div class="sidebar-section-title">Filtros</div>', unsafe_allow_html=True)
produto_sel = st.sidebar.selectbox("Produto", produtos, index=0 if modo == "Upload CSV/Excel" else min(1, len(produtos)-1))
regiao_sel = st.sidebar.selectbox("Região", regioes, index=0 if modo == "Upload CSV/Excel" else min(1, len(regioes)-1))

st.sidebar.markdown('<div class="sidebar-section-title">Previsão</div>', unsafe_allow_html=True)
horizonte = st.sidebar.slider("Semanas futuras", 1, 12, 4)
metodo_principal = st.sidebar.selectbox(
    "Modelo principal",
    ["Média móvel simples", "Média móvel ponderada", "Suavização exponencial simples", "Regressão linear simples", "Ingênuo"],
)

with st.sidebar.expander("Parâmetros do modelo", expanded=False):
    janela = st.slider("Janela da média móvel", 2, 6, 3)
    alfa = st.slider("Alfa da suavização exponencial", 0.05, 0.95, 0.30, 0.05)

with st.sidebar.expander("Cenário e estoque", expanded=False):
    capacidade = st.number_input("Capacidade produtiva semanal", min_value=0.0, value=0.0, step=10.0)
    estoque_atual = st.number_input("Estoque atual", min_value=0.0, value=0.0, step=10.0)
    estoque_seg = st.number_input("Estoque de segurança desejado", min_value=0.0, value=0.0, step=10.0)
    ajuste_whatif = st.slider("Ajuste da demanda prevista (%)", -50, 50, 0)
    meta_semanal = st.number_input("Meta semanal de demanda/produção", min_value=0.0, value=0.0, step=10.0)

st.sidebar.markdown('<div class="sidebar-section-title">Período</div>', unsafe_allow_html=True)
n = len(df_original)
semana_min = int(df_original["Semana"].min())
semana_max = int(df_original["Semana"].max())

# Correção de debug:
# O Streamlit não permite criar slider quando o valor mínimo é igual ao máximo.
# Isso pode acontecer quando existe apenas uma semana nos dados após entrada/upload.
# Nesse caso, o app fixa automaticamente o intervalo nessa única semana e segue exibindo
# o alerta de poucos dados mais abaixo, sem quebrar a execução.
if semana_min == semana_max:
    st.sidebar.info(f"Filtro de semanas fixo: apenas a semana {semana_min} foi informada.")
    intervalo_semanas = (semana_min, semana_max)
else:
    intervalo_semanas = st.sidebar.slider("Intervalo de semanas", semana_min, semana_max, (semana_min, semana_max))

# Registra alterações básicas no estado da sessão.
config_atual = f"{modo}|{produto_sel}|{regiao_sel}|{horizonte}|{metodo_principal}|{janela}|{alfa}|{ajuste_whatif}"
if st.session_state.get("config_anterior") != config_atual:
    st.session_state["historico_alteracoes"].append(
        {"Data/hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Alteração": config_atual}
    )
    st.session_state["config_anterior"] = config_atual

# Aplica filtros e agrega série.
df_filtrado = df_original[(df_original["Semana"] >= intervalo_semanas[0]) & (df_original["Semana"] <= intervalo_semanas[1])].copy()
serie_df = agregar_demanda(df_filtrado, produto_sel, regiao_sel)

if len(serie_df) < 2:
    st.warning("Após os filtros, há poucos dados. Ajuste os filtros ou informe mais semanas.")
    st.stop()
if len(serie_df) < 8:
    st.warning("O roteiro do professor recomenda entrada de 8 a 12 semanas no mínimo. Com menos dados, a previsão fica menos confiável.")

semanas = serie_df["Semana"].astype(int).tolist()
y = serie_df["Demanda"].astype(float).tolist()

# Cálculos principais.
previsao_base = prever(metodo_principal, y, horizonte, janela, alfa)
previsao_ajustada = [max(0.0, p * (1 + ajuste_whatif / 100)) for p in previsao_base]
semanas_futuras = list(range(max(semanas) + 1, max(semanas) + horizonte + 1))
df_previsao = pd.DataFrame({"Semana": semanas_futuras, "Previsão": np.round(previsao_ajustada, 2)})

df_backtest = backtest(metodo_principal, y, janela, alfa)
metricas = metricas_erro(df_backtest)
df_modelos = comparar_modelos(y, horizonte, janela, alfa)
validos = df_modelos.dropna(subset=["MAE"])
melhor_modelo = ""
if not validos.empty:
    melhor_modelo = str(validos.loc[validos["MAE"].idxmin(), "Modelo"])

tendencia, inclinacao, cv = analisar_tendencia(y)
df_anomalias = detectar_anomalias(serie_df)
qtd_anomalias = int(df_anomalias["Anomalia"].sum()) if "Anomalia" in df_anomalias else 0
df_ic = intervalo_confianca(previsao_ajustada, df_backtest["Erro absoluto"] if not df_backtest.empty else pd.Series(dtype=float))
df_ic.insert(0, "Semana", semanas_futuras)
df_sazonal = analisar_sazonalidade(serie_df)
media_hist = float(np.mean(y))
media_prev = float(np.mean(previsao_ajustada))
variacao_media = ((media_prev - media_hist) / media_hist * 100) if media_hist else 0
reposicao = max(0.0, sum(previsao_ajustada) + estoque_seg - estoque_atual)

# Indicadores comerciais quando a base tem preço/custo.
preco_medio = float(df_filtrado["Preco"].mean()) if "Preco" in df_filtrado.columns and not df_filtrado.empty else 0.0
custo_medio = float(df_filtrado["CustoUnitario"].mean()) if "CustoUnitario" in df_filtrado.columns and not df_filtrado.empty else 0.0
receita_prevista = float(sum(previsao_ajustada) * preco_medio) if preco_medio else 0.0
margem_prevista = float(sum(previsao_ajustada) * max(preco_medio - custo_medio, 0)) if preco_medio and custo_medio else 0.0

# Se a base simulada/upload trouxer estoque e capacidade do produto, usa como sugestão inicial.
if capacidade == 0 and "CapacidadeSemanal" in df_filtrado.columns and not df_filtrado.empty:
    capacidade = float(df_filtrado["CapacidadeSemanal"].max())
if estoque_atual == 0 and "EstoqueAtual" in df_filtrado.columns and not df_filtrado.empty:
    estoque_atual = float(df_filtrado["EstoqueAtual"].max())
if estoque_seg == 0 and "EstoqueMinimo" in df_filtrado.columns and not df_filtrado.empty:
    estoque_seg = float(df_filtrado["EstoqueMinimo"].max())
reposicao = max(0.0, sum(previsao_ajustada) + estoque_seg - estoque_atual)

recomendacao = gerar_recomendacao(
    tendencia=tendencia,
    previsoes=previsao_ajustada,
    media_hist=media_hist,
    capacidade=capacidade,
    estoque_atual=estoque_atual,
    estoque_seg=estoque_seg,
    anomalias=qtd_anomalias,
    melhor_modelo=melhor_modelo,
)

# Salva histórico de previsões quando o usuário clicar.
if st.sidebar.button("Salvar previsão no histórico"):
    st.session_state["historico_previsoes"].append(
        {
            "Data/hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "Produto": produto_sel,
            "Região": regiao_sel,
            "Modelo": metodo_principal,
            "Horizonte": horizonte,
            "Média prevista": round(media_prev, 2),
            "Reposição sugerida": round(reposicao, 2),
        }
    )
    st.sidebar.success("Previsão salva no histórico da sessão.")


# =========================================================
# DASHBOARD EXECUTIVO
# =========================================================
st.subheader("Dashboard Executivo")
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
col_kpi1.metric("Média histórica", f"{media_hist:.1f}")
col_kpi2.metric("Média prevista", f"{media_prev:.1f}", f"{variacao_media:.1f}%")
col_kpi3.metric("Tendência", tendencia.capitalize())
col_kpi4.metric("Precisão estimada", "N/D" if metricas["Precisão"] is None else f"{metricas['Precisão']:.1f}%")
col_kpi5.metric("Reposição sugerida", f"{reposicao:.0f}")
if receita_prevista > 0:
    cfin1, cfin2 = st.columns(2)
    cfin1.metric("Receita prevista no horizonte", f"R$ {receita_prevista:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    cfin2.metric("Margem bruta estimada", f"R$ {margem_prevista:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

if capacidade > 0 and max(previsao_ajustada) > capacidade:
    st.markdown(f"<div class='alerta'><strong>Alerta inteligente:</strong> a previsão máxima ({max(previsao_ajustada):.0f}) supera a capacidade semanal ({capacidade:.0f}).</div>", unsafe_allow_html=True)
if qtd_anomalias > 0:
    st.markdown(f"<div class='alerta'><strong>Alerta inteligente:</strong> foram encontradas {qtd_anomalias} possível(is) anomalia(s) no histórico.</div>", unsafe_allow_html=True)
if capacidade <= 0 and meta_semanal <= 0 and estoque_atual <= 0:
    st.markdown("<div class='ok'><strong>Nota:</strong> informe capacidade, meta e estoque na lateral para obter recomendações mais completas.</div>", unsafe_allow_html=True)


# =========================================================
# ABAS DO APP
# =========================================================
abas = st.tabs(
    [
        "Dados",
        "Previsão",
        "Modelos e Erros",
        "Cenários e Estoque",
        "Análises Avançadas",
        "Regiões e Produtos",
        "Recomendações",
        "Projeto e Estudos",
        "Históricos e Exportação",
    ]
)

with abas[0]:
    st.subheader("Entrada de dados e filtros")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Dados originais**")
        st.dataframe(df_original, width="stretch", hide_index=True)
    with c2:
        st.markdown("**Série usada na previsão**")
        st.dataframe(serie_df, width="stretch", hide_index=True)

    if modo == "Comércio cadastrado":
        st.markdown("**Produtos cadastrados**")
        cols_cadastro = [c for c in ["Produto", "Categoria", "Preco", "CustoUnitario", "EstoqueAtual", "EstoqueMinimo", "CapacidadeSemanal"] if c in df_original.columns]
        cadastro = df_original[cols_cadastro].drop_duplicates().sort_values("Produto")
        st.dataframe(cadastro, width="stretch", hide_index=True)

    st.markdown("**Validação dos dados**")
    st.write(f"Semanas usadas: {len(serie_df)}. Produto filtrado: {produto_sel}. Região filtrada: {regiao_sel}.")
    if len(serie_df) >= 8:
        st.success("Quantidade de semanas compatível com o escopo mínimo recomendado no roteiro.")
    else:
        st.warning("Quantidade de semanas abaixo do recomendado no roteiro; use o resultado com cautela.")

with abas[1]:
    st.subheader("Previsão multissemanal e intervalo de confiança")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df_previsao, width="stretch", hide_index=True)
    with c2:
        st.dataframe(df_ic.round(2), width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie_df["Semana"], y=serie_df["Demanda"], mode="lines+markers", name="Demanda real"))
    fig.add_trace(go.Scatter(x=df_previsao["Semana"], y=df_previsao["Previsão"], mode="lines+markers", name="Demanda prevista"))
    fig.add_trace(go.Scatter(x=df_ic["Semana"], y=df_ic["Limite superior"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=df_ic["Semana"],
            y=df_ic["Limite inferior"],
            mode="lines",
            fill="tonexty",
            name="Intervalo de confiança aproximado",
            line=dict(width=0),
        )
    )
    fig.update_layout(xaxis_title="Semana", yaxis_title="Demanda", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Explicação da previsão**")
    st.write(
        f"O método principal selecionado foi {metodo_principal}. O aplicativo calculou {horizonte} semana(s) futura(s). "
        f"A tendência histórica foi classificada como {tendencia}, com variação média prevista de {variacao_media:.1f}% em relação à média histórica."
    )

with abas[2]:
    st.subheader("Comparação de modelos, precisão e análise de erros")
    st.dataframe(df_modelos, width="stretch", hide_index=True)
    if melhor_modelo:
        st.success(f"Modelo com menor MAE no histórico: {melhor_modelo}.")
    st.info("Menor erro histórico não garante melhor previsão futura; serve como apoio à decisão.")

    if not df_backtest.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Comparação entre demanda prevista e real**")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=df_backtest["Semana"], y=df_backtest["Real"], mode="lines+markers", name="Real"))
            fig_bt.add_trace(go.Scatter(x=df_backtest["Semana"], y=df_backtest["Previsto"], mode="lines+markers", name="Previsto"))
            fig_bt.update_layout(xaxis_title="Semana", yaxis_title="Demanda")
            st.plotly_chart(fig_bt, width="stretch")
        with c2:
            st.markdown("**Análise de erros**")
            fig_err = px.bar(df_backtest, x="Semana", y="Erro", title="Erro por semana")
            st.plotly_chart(fig_err, width="stretch")
        st.dataframe(df_backtest.round(2), width="stretch", hide_index=True)
    else:
        st.warning("Ainda não há dados suficientes para backtest do método selecionado.")

with abas[3]:
    st.subheader("Simulação What-if, previsão de estoque e painel de cenários")
    base_sem_ajuste = pd.DataFrame({"Semana": semanas_futuras, "Previsão sem ajuste": prever(metodo_principal, y, horizonte, janela, alfa)})
    cenarios = pd.DataFrame(
        {
            "Semana": semanas_futuras,
            "Conservador (-10%)": [p * 0.90 for p in previsao_base],
            "Base": previsao_base,
            "Otimista (+10%)": [p * 1.10 for p in previsao_base],
            "What-if escolhido": previsao_ajustada,
        }
    )
    st.dataframe(cenarios.round(2), width="stretch", hide_index=True)
    fig_cen = px.line(cenarios, x="Semana", y=["Conservador (-10%)", "Base", "Otimista (+10%)", "What-if escolhido"], markers=True)
    fig_cen.update_layout(yaxis_title="Demanda prevista")
    st.plotly_chart(fig_cen, width="stretch")

    estoque = []
    saldo = estoque_atual
    producao_planejada = capacidade if capacidade > 0 else media_prev
    for sem, dem in zip(semanas_futuras, previsao_ajustada):
        saldo = saldo + producao_planejada - dem
        estoque.append({"Semana": sem, "Produção planejada": producao_planejada, "Demanda prevista": dem, "Estoque projetado": saldo})
    df_estoque = pd.DataFrame(estoque)
    st.markdown("**Previsão de estoque**")
    st.dataframe(df_estoque.round(2), width="stretch", hide_index=True)
    st.write(f"Sugestão de reposição/produção total no horizonte: **{reposicao:.0f} unidades**.")

    if meta_semanal > 0:
        desempenho = (media_prev / meta_semanal * 100) if meta_semanal else 0
        st.metric("Meta e desempenho", f"{desempenho:.1f}%", help="Média prevista dividida pela meta semanal informada.")

with abas[4]:
    st.subheader("Análise de tendência, sazonalidade, anomalias e sensibilidade")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Análise de tendência**")
        fig_tend = go.Figure()
        fig_tend.add_trace(go.Scatter(x=serie_df["Semana"], y=serie_df["Demanda"], mode="markers+lines", name="Demanda"))
        x_line = np.array(serie_df["Semana"], dtype=float)
        if len(x_line) >= 2:
            coef = np.polyfit(x_line, np.array(serie_df["Demanda"], dtype=float), 1)
            y_line = coef[0] * x_line + coef[1]
            fig_tend.add_trace(go.Scatter(x=serie_df["Semana"], y=y_line, mode="lines", name="Linha de tendência"))
        fig_tend.update_layout(title="Tendência histórica", xaxis_title="Semana", yaxis_title="Demanda")
        st.plotly_chart(fig_tend, width="stretch")
        st.write(f"Classificação: **{tendencia}**. Inclinação aproximada: {inclinacao:.2f} unidade(s) por semana.")
    with c2:
        st.markdown("**Sazonalidade simples por ciclo de 4 semanas**")
        st.dataframe(df_sazonal.round(2), width="stretch", hide_index=True)
        fig_saz = px.bar(df_sazonal, x="Ciclo de 4 semanas", y="Demanda média", title="Sazonalidade simplificada")
        st.plotly_chart(fig_saz, width="stretch")

    st.markdown("**Detecção de anomalias**")
    st.dataframe(df_anomalias.round(2), width="stretch", hide_index=True)
    if qtd_anomalias == 0:
        st.success("Nenhuma anomalia forte foi identificada pelo critério de Z-score.")

    st.markdown("**Análise de sensibilidade**")
    sensibilidades = pd.DataFrame(
        {
            "Cenário": ["-20%", "-10%", "Base", "+10%", "+20%"],
            "Média prevista": [np.mean([p * f for p in previsao_base]) for f in [0.8, 0.9, 1.0, 1.1, 1.2]],
            "Reposição sugerida": [max(0.0, sum([p * f for p in previsao_base]) + estoque_seg - estoque_atual) for f in [0.8, 0.9, 1.0, 1.1, 1.2]],
        }
    )
    st.dataframe(sensibilidades.round(2), width="stretch", hide_index=True)

    imp = feature_importance(df_filtrado)
    st.markdown("**Importância das variáveis**")
    if imp is not None:
        st.dataframe(imp.round(4), width="stretch", hide_index=True)
        fig_imp = px.bar(imp, x="Variável", y="Importância", title="Feature Importance")
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("Para calcular importância das variáveis, envie planilha com colunas como Promocao, Feriado, Evento ou Preco e pelo menos 8 linhas.")

with abas[5]:
    st.subheader("Mapa de demanda por região, ranking de produtos e heatmap")
    if "Regiao" in df_original.columns:
        reg = df_filtrado.groupby("Regiao", as_index=False)["Demanda"].sum().sort_values("Demanda", ascending=False)
        st.markdown("**Mapa de demanda por região**")
        st.dataframe(reg, width="stretch", hide_index=True)
        fig_reg = px.bar(reg, x="Regiao", y="Demanda", title="Demanda total por região")
        st.plotly_chart(fig_reg, width="stretch")

    prod = df_filtrado.groupby("Produto", as_index=False)["Demanda"].sum().sort_values("Demanda", ascending=False)
    st.markdown("**Ranking de produtos**")
    st.dataframe(prod, width="stretch", hide_index=True)
    fig_prod = px.bar(prod, x="Produto", y="Demanda", title="Ranking de produtos por demanda")
    st.plotly_chart(fig_prod, width="stretch")

    st.markdown("**Heatmap de demanda**")
    heat = df_filtrado.pivot_table(index="Produto", columns="Semana", values="Demanda", aggfunc="sum", fill_value=0)
    if not heat.empty:
        fig_heat = px.imshow(heat, aspect="auto", labels=dict(x="Semana", y="Produto", color="Demanda"))
        st.plotly_chart(fig_heat, width="stretch")

    st.markdown("**Análise de promoções, eventos e feriados**")
    cols_contexto = [c for c in ["Promocao", "Evento", "Feriado"] if c in df_filtrado.columns]
    if cols_contexto:
        for c in cols_contexto:
            aux = df_filtrado.groupby(c, as_index=False)["Demanda"].mean().rename(columns={"Demanda": "Demanda média"})
            st.write(f"Impacto médio por {c}:")
            st.dataframe(aux.round(2), width="stretch", hide_index=True)
    else:
        st.info("Para analisar promoções, eventos e feriados, envie uma planilha com essas colunas.")

with abas[6]:
    st.subheader("Recomendações inteligentes")
    st.markdown("**Recomendação Gerencial**")
    st.write(recomendacao)

    st.markdown("**Insights automáticos**")
    insights = []
    if tendencia == "crescente":
        insights.append("A demanda cresce ao longo das semanas; revise capacidade e fornecedores.")
    if tendencia == "decrescente":
        insights.append("A demanda cai ao longo das semanas; reduza risco de estoque parado.")
    if cv > 0.25:
        insights.append("A variação histórica é alta; investigue causas externas antes de decidir.")
    if metricas["MAPE"] is not None and metricas["MAPE"] > 20:
        insights.append("O erro percentual está alto; use a previsão com cautela e avalie outro método.")
    if capacidade > 0 and media_prev > capacidade:
        insights.append("A média prevista supera a capacidade produtiva informada.")
    if not insights:
        insights.append("Os indicadores não apontam alerta crítico, mas a previsão deve ser monitorada semanalmente.")
    for item in insights:
        st.write(f"- {item}")

    st.markdown("**Três recomendações práticas**")
    st.markdown(recomendacoes_praticas_markdown())

    st.markdown("**Limitações do aplicativo**")
    st.write(
        "O app utiliza métodos de previsão para apoio à decisão. Ele não substitui a análise do gestor, "
        "não captura sazonalidade complexa automaticamente e depende da qualidade dos dados informados."
    )

with abas[7]:
    st.subheader("Projeto, requisitos, variáveis e validação")
    st.markdown(resumo_estudo_markdown())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Requisitos funcionais e não funcionais**")
        st.dataframe(tabela_requisitos_app(), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Variáveis recomendadas para coleta**")
        st.dataframe(tabela_variaveis_app(), width="stretch", hide_index=True)

    st.markdown("**Plano de testes integrado**")
    st.dataframe(tabela_testes_app(), width="stretch", hide_index=True)

    st.markdown("**Recomendações práticas para o gestor da produção**")
    st.markdown(recomendacoes_praticas_markdown())

    st.markdown("**Limitações e fatores que reduzem a precisão**")
    st.write("Modelos simples podem perder precisão em sazonalidade forte, rupturas de mercado, dados incompletos, promoções inesperadas, falhas operacionais, atrasos de fornecedores e mudanças no comportamento do consumidor. Para reduzir esses riscos, recomenda-se atualizar dados semanalmente, comparar modelos, acompanhar erros, registrar eventos e usar estoque de segurança.")

    st.markdown("**Uso da IA generativa no projeto**")
    st.write("A IA auxiliou na estruturação do problema, requisitos, geração do código, debug, testes, documentação e melhoria da interface. A análise humana continua necessária para validar dados, interpretar resultados, reconhecer limitações e tomar decisões gerenciais.")

with abas[8]:
    st.subheader("Histórico de previsões, histórico de alterações e exportação de relatórios")
    st.markdown("**Histórico de previsões salvas**")
    hist_prev = pd.DataFrame(st.session_state["historico_previsoes"])
    if hist_prev.empty:
        st.info("Clique em 'Salvar previsão no histórico' na barra lateral para registrar uma previsão.")
    else:
        st.dataframe(hist_prev, width="stretch", hide_index=True)

    st.markdown("**Histórico de alterações da sessão**")
    hist_alt = pd.DataFrame(st.session_state["historico_alteracoes"])
    st.dataframe(hist_alt.tail(20), width="stretch", hide_index=True)

    contexto = {
        "Produto": produto_sel,
        "Região": regiao_sel,
        "Método principal": metodo_principal,
        "Horizonte": f"{horizonte} semana(s)",
        "Tendência": tendencia,
        "Média histórica": f"{media_hist:.2f}",
        "Média prevista": f"{media_prev:.2f}",
        "Melhor modelo no backtest": melhor_modelo or "Não calculado",
    }
    tabelas = {
        "Dados históricos usados": serie_df,
        "Previsão futura": df_previsao,
        "Comparação de modelos": df_modelos,
    }
    relatorio = criar_relatorio_texto(contexto, tabelas, recomendacao)
    st.download_button("Exportar relatório em TXT", relatorio.encode("utf-8"), "relatorio_previsao_demanda.txt", "text/plain")
    st.download_button("Exportar previsão em CSV", df_previsao.to_csv(index=False).encode("utf-8"), "previsao_demanda.csv", "text/csv")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        serie_df.to_excel(writer, sheet_name="historico", index=False)
        df_previsao.to_excel(writer, sheet_name="previsao", index=False)
        df_modelos.to_excel(writer, sheet_name="modelos", index=False)
        df_ic.to_excel(writer, sheet_name="intervalo", index=False)
        df_anomalias.to_excel(writer, sheet_name="anomalias", index=False)
    st.download_button(
        "Exportar relatório em Excel",
        buffer.getvalue(),
        "relatorio_previsao_demanda.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption(
    "Aplicativo acadêmico para Administração da Produção. Recursos: previsão multissemanal, comparação de modelos, KPIs, erros, anomalias, sazonalidade simples, estoque, cenários, ranking, heatmap, relatórios e recomendações gerenciais."
)
