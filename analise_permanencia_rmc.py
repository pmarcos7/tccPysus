from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utils import MUNICIPIOS_RMC_NORMALIZADOS, normalizar_nome



RAIZ = Path(__file__).resolve().parent
ARQUIVO_DADOS = (
    RAIZ / "TCC_Datasus" / "data" / "processed" / "gold_resp_ml.parquet"
)
PASTA_RESULTADOS = RAIZ / "resultados_permanencia"

COLUNAS_CATEGORICAS = [
    "SEXO",
    "FAIXA_ETARIA",
    "ESTACAO",
    "SUBCATEGORIA_CID",
    "TIPO_ATENDIMENTO",
]
COLUNA_MUNICIPIO = "NOME_MUNIC_ATENDIMENTO"
MINIMO_CASOS_POR_GRUPO = 30


def criar_modelo(categoricas):
    """Cria o mesmo modelo usando as variáveis categóricas recebidas."""
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
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calcular_metricas(nome, observado, previsto):
    return {
        "modelo": nome,
        "MAE_dias": mean_absolute_error(observado, previsto),
        "RMSE_dias": mean_squared_error(observado, previsto) ** 0.5,
        "R2": r2_score(observado, previsto),
    }


def main():
    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(
            f"Não encontrei a base de dados:\n{ARQUIVO_DADOS}\n"
            "Confira se gold_resp_ml.parquet está na pasta processed."
        )

    df = pd.read_parquet(ARQUIVO_DADOS)

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

    faltantes = [coluna for coluna in colunas_necessarias if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltam colunas na base: {faltantes}")

    df = df[colunas_necessarias].copy()

    # A base também contém moradores da RMC atendidos fora da região. Esses
    # registros entram apenas na contagem de saídas, não no modelo de permanência.
    df = df[
        df[COLUNA_MUNICIPIO].map(normalizar_nome).isin(MUNICIPIOS_RMC_NORMALIZADOS)
    ].copy()
    if df.empty:
        raise ValueError(
            "Não há internações atendidas em municípios da RMC na base gold_resp_ml. "
            "Execute novamente o pipeline com o recorte geográfico atualizado."
        )

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
    df["DIAG_PRINC"] = (
        df["DIAG_PRINC"]
        .fillna("Desconhecido")
        .astype(str)
        .str.strip()
        .replace("", "Desconhecido")
    )

    # Mantém a divisão temporal do trabalho:
    # treino em 2022–2024 e avaliação final em 2025.
    treino = df[df["ANO_CMPT"].between(2022, 2024)].copy()
    teste = df[df["ANO_CMPT"] == 2025].copy()

    if treino.empty or teste.empty:
        raise ValueError(
            f"Treino ou teste sem dados: treino={len(treino)}, teste={len(teste)}. "
            "Confira os anos na coluna ANO_CMPT."
        )

    y_treino = treino["DIAS_PERM"]
    y_teste = teste["DIAS_PERM"]
    previsoes = {}

    # Referência: prever a média de dias do conjunto de treino.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(treino[["IDADE"]], y_treino)
    previsoes["Baseline: média"] = baseline.predict(teste[["IDADE"]])

    # Modelo sem município de atendimento.
    modelo_sem_municipio = criar_modelo(COLUNAS_CATEGORICAS)
    modelo_sem_municipio.fit(treino, y_treino)
    previsoes["Modelo sem município"] = modelo_sem_municipio.predict(teste)

    # Mesmo modelo, agora incluindo o município onde houve atendimento.
    modelo_com_municipio = criar_modelo(
        COLUNAS_CATEGORICAS + [COLUNA_MUNICIPIO]
    )
    modelo_com_municipio.fit(treino, y_treino)
    previsoes["Modelo com município"] = modelo_com_municipio.predict(teste)

    # Mantém a especificação do HistGradientBoosting já usada no trabalho.
    features_cat_hgbr = [
        "FAIXA_ETARIA",
        "ESTACAO",
        "SUBCATEGORIA_CID",
        "TIPO_ATENDIMENTO",
    ]
    features_hgbr = ["IDADE"] + features_cat_hgbr
    preparar_hgbr = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                features_cat_hgbr,
            )
        ],
        remainder="passthrough",
    )
    modelo_hgbr = Pipeline(
        steps=[
            ("preprocessor", preparar_hgbr),
            ("regressor", HistGradientBoostingRegressor(random_state=42)),
        ]
    )
    modelo_hgbr.fit(treino[features_hgbr], y_treino)
    previsoes["HistGradientBoostingRegressor"] = modelo_hgbr.predict(
        teste[features_hgbr]
    )

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    tabela_metricas = pd.DataFrame(
        [
            calcular_metricas(nome, y_teste, valores)
            for nome, valores in previsoes.items()
        ]
    )
    tabela_metricas.insert(1, "casos_treino", len(treino))
    tabela_metricas.insert(2, "casos_teste_2025", len(teste))
    tabela_metricas.to_csv(
        PASTA_RESULTADOS / "metricas_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Previsões detalhadas para cada internação de 2025.
    resultado = teste[
        [
            "ANO_CMPT",
            "IDADE",
            "SEXO",
            "FAIXA_ETARIA",
            "SUBCATEGORIA_CID",
            "DIAG_PRINC",
            "TIPO_ATENDIMENTO",
            COLUNA_MUNICIPIO,
            "DIAS_PERM",
        ]
    ].copy()

    resultado = resultado.rename(columns={"DIAS_PERM": "DIAS_OBSERVADOS"})
    resultado["PREVISTO_BASELINE"] = previsoes["Baseline: média"]
    resultado["PREVISTO_SEM_MUNICIPIO"] = previsoes["Modelo sem município"]
    resultado["PREVISTO_COM_MUNICIPIO"] = previsoes["Modelo com município"]
    resultado["PREVISTO_HGBR"] = previsoes["HistGradientBoostingRegressor"]

    resultado["ERRO_ABSOLUTO_BASELINE"] = (
        resultado["DIAS_OBSERVADOS"] - resultado["PREVISTO_BASELINE"]
    ).abs()
    resultado["ERRO_ABSOLUTO_SEM_MUNICIPIO"] = (
        resultado["DIAS_OBSERVADOS"] - resultado["PREVISTO_SEM_MUNICIPIO"]
    ).abs()

    resultado["ERRO_ABSOLUTO_COM_MUNICIPIO"] = (
        resultado["DIAS_OBSERVADOS"] - resultado["PREVISTO_COM_MUNICIPIO"]
    ).abs()
    resultado["ERRO_ABSOLUTO_HGBR"] = (
        resultado["DIAS_OBSERVADOS"] - resultado["PREVISTO_HGBR"]
    ).abs()

    resultado["DIFERENCA_OBSERVADO_ESPERADO"] = (
        resultado["DIAS_OBSERVADOS"] - resultado["PREVISTO_SEM_MUNICIPIO"]
    )

    resultado.to_csv(
        PASTA_RESULTADOS / "previsoes_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Avalia em quais perfis o modelo reduz o erro frente à média do treino.
    metricas_por_grupo = (
        resultado.groupby(
            ["SUBCATEGORIA_CID", "FAIXA_ETARIA", "TIPO_ATENDIMENTO"],
            dropna=False,
        )
        .agg(
            internacoes=("DIAS_OBSERVADOS", "size"),
            mediana_observada=("DIAS_OBSERVADOS", "median"),
            p90_observado=("DIAS_OBSERVADOS", lambda valores: valores.quantile(0.90)),
            mae_baseline=("ERRO_ABSOLUTO_BASELINE", "mean"),
            mae_sem_municipio=("ERRO_ABSOLUTO_SEM_MUNICIPIO", "mean"),
            mae_com_municipio=("ERRO_ABSOLUTO_COM_MUNICIPIO", "mean"),
            mae_hgbr=("ERRO_ABSOLUTO_HGBR", "mean"),
        )
        .reset_index()
    )
    metricas_por_grupo = metricas_por_grupo[
        metricas_por_grupo["internacoes"] >= MINIMO_CASOS_POR_GRUPO
    ].copy()
    metricas_por_grupo["ganho_mae_sem_municipio"] = (
        metricas_por_grupo["mae_baseline"] - metricas_por_grupo["mae_sem_municipio"]
    )
    metricas_por_grupo["ganho_mae_com_municipio"] = (
        metricas_por_grupo["mae_baseline"] - metricas_por_grupo["mae_com_municipio"]
    )
    metricas_por_grupo["ganho_mae_hgbr"] = (
        metricas_por_grupo["mae_baseline"] - metricas_por_grupo["mae_hgbr"]
    )
    metricas_por_grupo.to_csv(
        PASTA_RESULTADOS / "metricas_por_grupo_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Comparação por doença, faixa etária e município de atendimento.
    comparacao = (
        resultado.groupby(
            [
                "SUBCATEGORIA_CID",
                "FAIXA_ETARIA",
                COLUNA_MUNICIPIO,
            ],
            dropna=False,
        )
        .agg(
            internacoes=("DIAS_OBSERVADOS", "size"),
            media_observada=("DIAS_OBSERVADOS", "mean"),
            mediana_observada=("DIAS_OBSERVADOS", "median"),
            primeiro_quartil=(
                "DIAS_OBSERVADOS",
                lambda valores: valores.quantile(0.25),
            ),
            terceiro_quartil=(
                "DIAS_OBSERVADOS",
                lambda valores: valores.quantile(0.75),
            ),
            media_prevista=("PREVISTO_SEM_MUNICIPIO", "mean"),
            diferenca_observado_esperado=(
                "DIFERENCA_OBSERVADO_ESPERADO",
                "mean",
            ),
            erro_medio_sem_municipio=(
                "ERRO_ABSOLUTO_SEM_MUNICIPIO",
                "mean",
            ),
            erro_medio_com_municipio=(
                "ERRO_ABSOLUTO_COM_MUNICIPIO",
                "mean",
            ),
        )
        .reset_index()
    )

    # Evita destacar grupos com poucos casos.
    comparacao = comparacao[
        comparacao["internacoes"] >= MINIMO_CASOS_POR_GRUPO
    ].copy()

    comparacao = comparacao.sort_values(
        [
            "SUBCATEGORIA_CID",
            "FAIXA_ETARIA",
            "diferenca_observado_esperado",
        ]
    )

    comparacao.to_csv(
        PASTA_RESULTADOS / "comparacao_por_municipio_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDesempenho dos modelos no teste de 2025:")
    print(tabela_metricas.round(3).to_string(index=False))
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")
    print(
        "\nA diferença observado-esperado é uma associação nos dados. "
        "Ela não prova que uma cidade cause uma internação mais curta ou longa."
    )


if __name__ == "__main__":
    main()
