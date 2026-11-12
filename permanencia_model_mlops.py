"""
Script de MLOps integrado ao projeto TCC DATASUS (RMC).
Coloque este arquivo na raiz do repositório `tccPysus` (mesma pasta de `main.py` e `utils.py`).

O que este script faz:
1. Carrega a base real do TCC (`TCC_Datasus/data/processed/gold_resp_ml.parquet`)
   ou utiliza gerador de contingência (`--demo-fallback`) caso os arquivos brutos não estejam presentes.
2. Treina 4 abordagens (Baseline Média, HistGradientBoosting, Random Forest sem Município
   e Random Forest com Município) particionando temporalmente em 2022-2024 (treino) e 2025 (teste).
3. Grava no MLflow hiperparâmetros, métricas (MAE, RMSE, R2) e serializa os pipelines
   com 'cloudpickle' para evitar falhas de tipos não confiáveis no skops.
4. Registra formalmente o modelo vencedor no Model Registry sob o nome 'Permanencia_RMC_Regressor'
   e vincula o alias '@champion' para consumo dinâmico na API.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from utils import MUNICIPIOS_RMC_NORMALIZADOS, MUNICIPIOS_RMC_NOMES, normalizar_nome


RAIZ = Path(__file__).resolve().parent
ARQUIVO_DADOS = RAIZ / "TCC_Datasus" / "data" / "processed" / "gold_resp_ml.parquet"
PASTA_RESULTADOS = RAIZ / "resultados_permanencia"

TRACKING_URI = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
EXPERIMENTO_NOME = "SUS_RMC_Permanencia_Hospitalar"
MODELO_REGISTRO_NOME = "Permanencia_RMC_Regressor"

COLUNAS_CATEGORICAS = [
    "SEXO",
    "FAIXA_ETARIA",
    "ESTACAO",
    "SUBCATEGORIA_CID",
    "TIPO_ATENDIMENTO",
]
COLUNA_MUNICIPIO = "NOME_MUNIC_ATENDIMENTO"
MINIMO_CASOS_POR_GRUPO = 30


def criar_modelo_rf(categoricas, n_estimators=300, min_samples_leaf=5):
    transformar = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline(
                    steps=[
                        (
                            "preencher",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Desconhecido",
                            ),
                        ),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categoricas,
            ),
            (
                "idade",
                SimpleImputer(strategy="median"),
                ["IDADE"],
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preparar_dados", transformar),
            (
                "modelo",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def criar_modelo_hgbr(max_iter=100, learning_rate=0.1):
    features_cat_hgbr = [
        "FAIXA_ETARIA",
        "ESTACAO",
        "SUBCATEGORIA_CID",
        "TIPO_ATENDIMENTO",
    ]
    preparar_hgbr = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                features_cat_hgbr,
            ),
            (
                "num",
                SimpleImputer(strategy="median"),
                ["IDADE"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preparar_hgbr),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    random_state=42,
                ),
            ),
        ]
    )


def criar_modelo_baseline():
    transformar = ColumnTransformer(
        transformers=[("idade", SimpleImputer(strategy="median"), ["IDADE"])],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preparar_dados", transformar),
            ("modelo", DummyRegressor(strategy="mean")),
        ]
    )


def gerar_base_sintetica_contingencia(n_amostras=6000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    anos = rng.choice([2022, 2023, 2024, 2025], size=n_amostras, p=[0.25, 0.25, 0.25, 0.25])
    idades = rng.integers(0, 95, size=n_amostras)
    faixas = [
        "Crianca" if i <= 11 else "Adolescente" if i <= 17 else "Adulto" if i <= 59 else "Idoso"
        for i in idades
    ]
    sexos = rng.choice(["1", "3"], size=n_amostras)
    estacoes = rng.choice(["Inverno", "Outras"], size=n_amostras, p=[0.45, 0.55])
    subcategorias = rng.choice(
        ["Influenza/Pneumonia", "DPOC/Asma", "IRA Superior", "IRA Aguda", "Outras Respiratorias"],
        size=n_amostras,
        p=[0.40, 0.25, 0.10, 0.15, 0.10],
    )
    tipos = rng.choice(["Urgencia", "Eletiva"], size=n_amostras, p=[0.85, 0.15])
    municipios = rng.choice(MUNICIPIOS_RMC_NOMES, size=n_amostras)
    cnes = rng.choice([2079798, 2080575, 2081490, 2755181, 2078015], size=n_amostras)

    base_dias = (
        2.5
        + (idades * 0.05)
        + np.where(np.array(subcategorias) == "Influenza/Pneumonia", 2.8, 0.5)
        + np.where(np.array(tipos) == "Urgencia", 1.4, 0.0)
        + np.where(np.array(estacoes) == "Inverno", 0.8, 0.0)
        + rng.normal(0, 2.0, size=n_amostras)
    )
    dias_perm = np.clip(np.round(base_dias), 1, 60).astype(int)

    return pd.DataFrame({
        "ANO_CMPT": anos,
        "IDADE": idades,
        "SEXO": sexos,
        "FAIXA_ETARIA": faixas,
        "ESTACAO": estacoes,
        "SUBCATEGORIA_CID": subcategorias,
        "DIAG_PRINC": "J189",
        "TIPO_ATENDIMENTO": tipos,
        "NOME_MUNIC_ATENDIMENTO": municipios,
        "CNES": cnes,
        "DIAS_PERM": dias_perm,
    })


def carregar_e_preparar_dados(usar_fallback=False):
    if ARQUIVO_DADOS.exists():
        print(f"[Dados] Carregando base oficial: {ARQUIVO_DADOS}")
        df = pd.read_parquet(ARQUIVO_DADOS)
    elif usar_fallback:
        print("[Aviso] Usando dados sinteticos de contingencia (--demo-fallback).")
        df = gerar_base_sintetica_contingencia()
    else:
        raise FileNotFoundError(
            f"Nao encontrei a base de dados em:\n{ARQUIVO_DADOS}\n"
            "Coloque gold_resp_ml.parquet na pasta processed ou execute com --demo-fallback."
        )

    colunas_necessarias = [
        "IDADE",
        "SEXO",
        "FAIXA_ETARIA",
        "ESTACAO",
        "SUBCATEGORIA_CID",
        "DIAG_PRINC",
        "TIPO_ATENDIMENTO",
        "ANO_CMPT",
        "DIAS_PERM",
        COLUNA_MUNICIPIO,
    ]
    if "CNES" in df.columns:
        colunas_necessarias.append("CNES")

    df = df[colunas_necessarias].copy()
    df = df[
        df[COLUNA_MUNICIPIO].map(normalizar_nome).isin(MUNICIPIOS_RMC_NORMALIZADOS)
    ].copy()

    for coluna in ["IDADE", "ANO_CMPT", "DIAS_PERM"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    df = df.dropna(subset=["ANO_CMPT", "DIAS_PERM"])

    todas_categoricas = COLUNAS_CATEGORICAS + [COLUNA_MUNICIPIO]
    for coluna in todas_categoricas:
        df[coluna] = (
            df[coluna]
            .fillna("Desconhecido")
            .astype(str)
            .str.strip()
            .replace("", "Desconhecido")
        )

    treino = df[df["ANO_CMPT"].between(2022, 2024)].copy()
    teste = df[df["ANO_CMPT"] == 2025].copy()
    return treino, teste


def main():
    parser = argparse.ArgumentParser(description="Esteira MLOps - Permanencia Hospitalar RMC")
    parser.add_argument(
        "--demo-fallback",
        action="store_true",
        help="Usa dados sinteticos caso gold_resp_ml.parquet nao esteja disponivel.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO_NOME)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    treino, teste = carregar_e_preparar_dados(usar_fallback=args.demo_fallback)
    features_entrada = ["IDADE"] + COLUNAS_CATEGORICAS + [COLUNA_MUNICIPIO]

    X_treino = treino[features_entrada]
    y_treino = treino["DIAS_PERM"]
    X_teste = teste[features_entrada]
    y_teste = teste["DIAS_PERM"]

    experimentos = [
        (
            "Baseline: média",
            "01_Baseline_Media",
            criar_modelo_baseline(),
            {"algoritmo": "DummyRegressor", "strategy": "mean", "usa_municipio": False},
        ),
        (
            "HistGradientBoostingRegressor",
            "02_HistGradientBoosting_RMC",
            criar_modelo_hgbr(max_iter=100, learning_rate=0.1),
            {"algoritmo": "HistGradientBoostingRegressor", "max_iter": 100, "learning_rate": 0.1, "usa_municipio": False},
        ),
        (
            "Modelo sem município",
            "03_RandomForest_Sem_Municipio",
            criar_modelo_rf(COLUNAS_CATEGORICAS, n_estimators=300, min_samples_leaf=5),
            {"algoritmo": "RandomForestRegressor", "n_estimators": 300, "min_samples_leaf": 5, "usa_municipio": False},
        ),
        (
            "Modelo com município",
            "04_RandomForest_Com_Municipio",
            criar_modelo_rf(COLUNAS_CATEGORICAS + [COLUNA_MUNICIPIO], n_estimators=300, min_samples_leaf=5),
            {"algoritmo": "RandomForestRegressor", "n_estimators": 300, "min_samples_leaf": 5, "usa_municipio": True},
        ),
    ]

    previsoes = {}
    linhas_metricas = []
    melhor_mae = float("inf")
    melhor_run_id = None
    melhor_nome = None

    print("\n=== Iniciando Rastreamento de Experimentos no MLflow ===")
    for nome_legivel, nome_run, modelo_pipeline, hiperparametros in experimentos:
        with mlflow.start_run(run_name=nome_run) as run:
            print(f"-> Treinando e rastreando: {nome_legivel} (Run ID: {run.info.run_id[:8]}...)")

            mlflow.log_params(hiperparametros)
            mlflow.log_param("recorte_geografico", "20_Municipios_RMC")
            mlflow.log_param("periodo_treino", "2022-2024")
            mlflow.log_param("periodo_teste", "2025")
            mlflow.log_param("casos_treino", len(X_treino))
            mlflow.log_param("casos_teste_2025", len(X_teste))

            modelo_pipeline.fit(X_treino, y_treino)
            preds = modelo_pipeline.predict(X_teste)
            previsoes[nome_legivel] = preds

            mae = float(mean_absolute_error(y_teste, preds))
            rmse = float(mean_squared_error(y_teste, preds) ** 0.5)
            r2 = float(r2_score(y_teste, preds))

            mlflow.log_metrics({
                "MAE_dias": round(mae, 4),
                "RMSE_dias": round(rmse, 4),
                "R2": round(r2, 4),
            })

            assinatura = infer_signature(X_teste.head(5), preds[:5])
            mlflow.sklearn.log_model(
                sk_model=modelo_pipeline,
                name="modelo_permanencia",
                signature=assinatura,
                input_example=X_teste.head(3),
                serialization_format="cloudpickle",
            )

            linhas_metricas.append({
                "modelo": nome_legivel,
                "casos_treino": len(X_treino),
                "casos_teste_2025": len(X_teste),
                "MAE_dias": mae,
                "RMSE_dias": rmse,
                "R2": r2,
                "mlflow_run_id": run.info.run_id,
            })

            if mae < melhor_mae:
                melhor_mae = mae
                melhor_run_id = run.info.run_id
                melhor_nome = nome_legivel

    tabela_metricas = pd.DataFrame(linhas_metricas)
    arq_metricas = PASTA_RESULTADOS / "metricas_modelos.csv"
    tabela_metricas.drop(columns=["mlflow_run_id"]).to_csv(
        arq_metricas, index=False, encoding="utf-8-sig"
    )

    print("\n=== Governanca: Registrando Modelo Campeao no Model Registry ===")
    model_uri = f"runs:/{melhor_run_id}/modelo_permanencia"
    versao = mlflow.register_model(model_uri=model_uri, name=MODELO_REGISTRO_NOME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODELO_REGISTRO_NOME,
        alias="champion",
        version=versao.version,
    )
    client.set_model_version_tag(
        name=MODELO_REGISTRO_NOME,
        version=versao.version,
        key="validacao_temporal",
        value="Treino_2022_2024_Teste_2025",
    )

    print(f"Modelo vencedor: {melhor_nome} (MAE = {melhor_mae:.3f} dias)")
    print(f"Registrado como: '{MODELO_REGISTRO_NOME}' | Versao: {versao.version} | Alias: @champion")
    print(f"\nTabela de Desempenho no Teste de 2025:")
    print(tabela_metricas[["modelo", "MAE_dias", "RMSE_dias", "R2"]].round(3).to_string(index=False))
    print("\nPara abrir o painel do MLflow no navegador, execute:")
    print("  python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")


if __name__ == "__main__":
    main()