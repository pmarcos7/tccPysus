"""Previsão de tempo de permanência para novos pacientes (ex.: 2026).

A função prever_demanda_2026() recebe as características de um ou mais pacientes
e devolve a estimativa de dias de internação. Usa apenas variáveis conhecidas
NA ADMISSÃO (idade, faixa etária, estação, subcategoria de CID e tipo de
atendimento); não usa a passagem pela UTI, que só é conhecida durante a
internação. O modelo de produção é treinado com todo o histórico disponível
(2022–2025) e persistido em disco para reuso.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from utils import MUNICIPIOS_RMC_NORMALIZADOS, normalizar_nome

RAIZ = Path(__file__).resolve().parent
ARQ_GOLD = RAIZ / "TCC_Datasus" / "data" / "processed" / "gold_resp_ml.parquet"
ARQ_MODELO = RAIZ / "TCC_Datasus" / "data" / "processed" / "modelo_permanencia_2026.joblib"

FEATURES_CATEGORICAS = ["FAIXA_ETARIA", "ESTACAO", "SUBCATEGORIA_CID", "TIPO_ATENDIMENTO"]
FEATURES_NUMERICAS = ["IDADE"]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS


def treinar_modelo_producao(caminho_gold=ARQ_GOLD, salvar=True):
    """Treina o modelo de produção com todo o histórico (2022–2025) e o salva."""
    df = pd.read_parquet(caminho_gold)
    df = df[
        df["NOME_MUNIC_ATENDIMENTO"].map(normalizar_nome).isin(MUNICIPIOS_RMC_NORMALIZADOS)
    ].copy()
    df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")
    df["DIAS_PERM"] = pd.to_numeric(df["DIAS_PERM"], errors="coerce")
    df = df.dropna(subset=["DIAS_PERM"])

    modelo = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                            FEATURES_CATEGORICAS,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("regressor", HistGradientBoostingRegressor(random_state=42)),
        ]
    )
    modelo.fit(df[FEATURES], df["DIAS_PERM"])
    if salvar:
        joblib.dump(modelo, ARQ_MODELO)
        print(f"Modelo de produção salvo em: {ARQ_MODELO}")
    return modelo


def prever_demanda_2026(dados, modelo=None):
    """Estima o tempo de permanência (dias) de novos pacientes.

    `dados` pode ser um dicionário (um paciente) ou um DataFrame (vários),
    contendo as colunas de FEATURES. Retorna um array com os dias previstos.
    """
    if modelo is None:
        modelo = joblib.load(ARQ_MODELO) if ARQ_MODELO.exists() else treinar_modelo_producao()
    if isinstance(dados, dict):
        dados = pd.DataFrame([dados])
    faltantes = [c for c in FEATURES if c not in dados.columns]
    if faltantes:
        raise ValueError(f"Faltam colunas do paciente: {faltantes}")
    return modelo.predict(dados[FEATURES])


if __name__ == "__main__":
    treinar_modelo_producao()
    exemplos = pd.DataFrame(
        [
            {
                "IDADE": 73,
                "FAIXA_ETARIA": "Idoso",
                "ESTACAO": "Inverno",
                "SUBCATEGORIA_CID": "Influenza/Pneumonia",
                "TIPO_ATENDIMENTO": "Urgencia",
            },
            {
                "IDADE": 23,
                "FAIXA_ETARIA": "Adulto",
                "ESTACAO": "Outras",
                "SUBCATEGORIA_CID": "IRA Aguda",
                "TIPO_ATENDIMENTO": "Urgencia",
            },
        ]
    )
    exemplos["DIAS_PREVISTOS"] = prever_demanda_2026(exemplos).round(1)
    print("\nExemplo de previsão para novos pacientes (2026):")
    print(exemplos.to_string(index=False))
