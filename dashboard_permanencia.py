from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import MUNICIPIOS_RMC_NORMALIZADOS, normalizar_nome


st.set_page_config(
    page_title="Internações respiratórias | RMC",
    page_icon="🏥",
    layout="wide",
)


def localizar_resultados() -> Path:
    """Find the project results folder when this file is placed at the repo root."""
    script_dir = Path(__file__).resolve().parent
    candidatos = [script_dir, *script_dir.parents]
    for pasta in candidatos:
        if (pasta / "resultados_permanencia").is_dir():
            return pasta / "resultados_permanencia"
    return script_dir / "resultados_permanencia"


PASTA_RESULTADOS = localizar_resultados()
PASTA_PROJETO = PASTA_RESULTADOS.parent
ARQUIVO_DESLOCAMENTOS = (
    PASTA_PROJETO / "TCC_Datasus" / "data" / "processed" / "gold_resp_eda.parquet"
)
ARQUIVO_FLUXO_SAIDAS = (
    PASTA_PROJETO / "TCC_Datasus" / "data" / "processed" / "gold_resp_fluxo.parquet"
)

@st.cache_data
def _ler_csv_em_cache(caminho: str, modificacao: float) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="utf-8-sig")


def carregar_csv(nome: str) -> pd.DataFrame:
    caminho = PASTA_RESULTADOS / nome
    if not caminho.is_file():
        return pd.DataFrame()
    return _ler_csv_em_cache(str(caminho), caminho.stat().st_mtime)


@st.cache_data
def _ler_parquet_em_cache(caminho: str, modificacao: float) -> pd.DataFrame:
    colunas = [
        "ANO_CMPT", "MUNIC_RES", "MUNIC_ATENDIMENTO",
        "NOME_MUNIC_RES", "NOME_MUNIC_ATENDIMENTO", "DIAG_PRINC",
        "SUBCATEGORIA_CID", "FAIXA_ETARIA", "TIPO_ATENDIMENTO",
        "DIAS_PERM", "MORTE",
    ]
    return pd.read_parquet(caminho, columns=colunas)


def carregar_deslocamentos(caminho: str) -> pd.DataFrame:
    arquivo = Path(caminho)
    if not arquivo.is_file():
        return pd.DataFrame()
    return _ler_parquet_em_cache(str(arquivo), arquivo.stat().st_mtime)


@st.cache_data
def _ler_fluxo_saidas_em_cache(caminho: str, modificacao: float) -> pd.DataFrame:
    return pd.read_parquet(caminho)


def carregar_fluxo_saidas(caminho: str) -> pd.DataFrame:
    arquivo = Path(caminho)
    if not arquivo.is_file():
        return pd.DataFrame()
    return _ler_fluxo_saidas_em_cache(str(arquivo), arquivo.stat().st_mtime)


st.title("Internações por doenças respiratórias na RMC")
st.caption("Modelos de previsão e comparação exploratória por estabelecimento, doença e faixa etária — ano de teste: 2025")

metricas = carregar_csv("metricas_modelos.csv")
hospitais = carregar_csv("comparacao_hospitais_2025.csv")
deslocamentos = carregar_deslocamentos(str(ARQUIVO_DESLOCAMENTOS))
fluxo_saidas = carregar_fluxo_saidas(str(ARQUIVO_FLUXO_SAIDAS))
metricas_por_grupo = carregar_csv("metricas_por_grupo_2025.csv")

if metricas.empty or hospitais.empty:
    st.error(
        "Não encontrei os arquivos de resultados. Coloque este script na pasta principal "
        "do projeto, ao lado da pasta `resultados_permanencia`, e confirme que nela estão "
        "`metricas_modelos.csv` e `comparacao_hospitais_2025.csv`."
    )
    st.code(str(PASTA_RESULTADOS))
    st.stop()

metricas = metricas.copy()
metricas["origem"] = "metricas_modelos.csv"
for coluna in ["MAE_dias", "RMSE_dias", "R2"]:
    metricas[coluna] = pd.to_numeric(metricas[coluna], errors="coerce")

hospitais = hospitais.copy()
if "NOME_MUNIC_ATENDIMENTO" in hospitais.columns:
    hospitais = hospitais[
        hospitais["NOME_MUNIC_ATENDIMENTO"]
        .map(normalizar_nome)
        .isin(MUNICIPIOS_RMC_NORMALIZADOS)
    ].copy()
for coluna in [
    "CNES",
    "N_INTERNACOES",
    "MEDIA_OBSERVADA_DIAS",
    "MEDIA_PREVISTA_DIAS",
    "DIF_OBSERVADO_PREVISTO_DIAS",
]:
    if coluna in hospitais.columns:
        hospitais[coluna] = pd.to_numeric(hospitais[coluna], errors="coerce")

aba_modelos, aba_hospitais, aba_perfil, aba_previsao_grupos, aba_deslocamentos, aba_metodo = st.tabs(
    [
        "📊 Modelos",
        "🏥 Hospitais",
        "📈 Perfil da permanência",
        "🎯 Previsão por grupo",
        "🧭 Deslocamentos",
        "ℹ️ Como interpretar",
    ]
)

with aba_modelos:
    st.subheader("Desempenho no teste de 2025")
    st.markdown(
        "MAE e RMSE medem o erro em dias; valores menores indicam previsões mais próximas. "
        "R² resume quanto da variação o modelo explica."
    )
    st.dataframe(
        metricas[["modelo", "casos_treino", "casos_teste_2025", "MAE_dias", "RMSE_dias", "R2", "origem"]]
        .rename(columns={
            "modelo": "Modelo",
            "casos_treino": "Internações no treino",
            "casos_teste_2025": "Internações no teste 2025",
            "MAE_dias": "MAE (dias)",
            "RMSE_dias": "RMSE (dias)",
            "R2": "R²",
            "origem": "Origem das métricas",
        })
        .round({"MAE (dias)": 3, "RMSE (dias)": 3, "R²": 3}),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        "Todas as métricas vêm da última execução de `analise_permanencia_rmc.py` e usam "
        "internações até 2024 para treino e 2025 para teste. Os modelos podem usar conjuntos "
        "diferentes de variáveis; portanto, a tabela compara as versões completas dos modelos, "
        "não apenas os algoritmos isoladamente."
    )
    grafico_metricas = metricas.melt(
        id_vars="modelo", value_vars=["MAE_dias", "RMSE_dias"],
        var_name="Métrica", value_name="Erro (dias)"
    )
    fig_metricas = px.bar(
        grafico_metricas,
        x="modelo",
        y="Erro (dias)",
        color="Métrica",
        barmode="group",
        title="Erros dos modelos no teste de 2025",
        labels={"modelo": "Modelo"},
    )
    st.plotly_chart(fig_metricas, use_container_width=True)

with aba_hospitais:
    st.subheader("Comparação por estabelecimento")
    st.caption(
        "A tabela contém grupos com pelo menos 30 internações, agrupados por CNES, "
        "doença e faixa etária."
    )

    filtros = st.columns(3)
    doencas = sorted(hospitais["SUBCATEGORIA_CID"].dropna().astype(str).unique())
    faixas = sorted(hospitais["FAIXA_ETARIA"].dropna().astype(str).unique())
    municipios = sorted(hospitais["NOME_MUNIC_ATENDIMENTO"].dropna().astype(str).unique())
    with filtros[0]:
        doenca = st.selectbox("Doença", ["Todas"] + doencas)
    with filtros[1]:
        faixa = st.selectbox("Faixa etária", ["Todas"] + faixas)
    with filtros[2]:
        municipio = st.selectbox("Município de atendimento", ["Todos"] + municipios)

    dados = hospitais.copy()
    if doenca != "Todas":
        dados = dados[dados["SUBCATEGORIA_CID"].astype(str) == doenca]
    if faixa != "Todas":
        dados = dados[dados["FAIXA_ETARIA"].astype(str) == faixa]
    if municipio != "Todos":
        dados = dados[dados["NOME_MUNIC_ATENDIMENTO"].astype(str) == municipio]

    if dados.empty:
        st.warning("Não há grupos para essa combinação de filtros.")
    else:
        cnes_count = dados["CNES"].nunique()
        casos_count = int(dados["N_INTERNACOES"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Estabelecimentos (CNES)", f"{cnes_count}")
        m2.metric("Grupos doença/faixa", f"{len(dados)}")
        m3.metric("Internações nos grupos", f"{casos_count:,}".replace(",", "."))

        dados_grafico = dados.sort_values("MEDIA_OBSERVADA_DIAS", ascending=False)
        dados_grafico["CNES_LABEL"] = dados_grafico["CNES"].fillna(0).astype(int).astype(str)
        fig_hospitais = px.scatter(
            dados_grafico,
            x="MEDIA_PREVISTA_DIAS",
            y="MEDIA_OBSERVADA_DIAS",
            size="N_INTERNACOES",
            color="DIF_OBSERVADO_PREVISTO_DIAS",
            hover_name="CNES_LABEL",
            hover_data={
                "NOME_MUNIC_ATENDIMENTO": True,
                "SUBCATEGORIA_CID": True,
                "FAIXA_ETARIA": True,
                "N_INTERNACOES": True,
                "DIF_OBSERVADO_PREVISTO_DIAS": ":.2f",
                "CNES_LABEL": False,
            },
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            labels={
                "MEDIA_PREVISTA_DIAS": "Média prevista (dias)",
                "MEDIA_OBSERVADA_DIAS": "Média observada (dias)",
                "DIF_OBSERVADO_PREVISTO_DIAS": "Observado − previsto (dias)",
                "N_INTERNACOES": "Internações",
            },
            title="Tempo observado versus previsto por grupo",
        )
        limite = max(
            dados["MEDIA_PREVISTA_DIAS"].max(),
            dados["MEDIA_OBSERVADA_DIAS"].max(),
        )
        fig_hospitais.add_shape(
            type="line", x0=0, y0=0, x1=limite, y1=limite,
            line={"dash": "dash", "color": "gray"},
        )
        st.plotly_chart(fig_hospitais, use_container_width=True)

        tabela = dados.sort_values(
            "DIF_OBSERVADO_PREVISTO_DIAS", ascending=False
        ).rename(columns={
            "CNES": "CNES",
            "NOME_MUNIC_ATENDIMENTO": "Município de atendimento",
            "SUBCATEGORIA_CID": "Doença",
            "FAIXA_ETARIA": "Faixa etária",
            "N_INTERNACOES": "Internações",
            "MEDIA_OBSERVADA_DIAS": "Média observada (dias)",
            "MEDIA_PREVISTA_DIAS": "Média prevista (dias)",
            "DIF_OBSERVADO_PREVISTO_DIAS": "Observado − previsto (dias)",
        })
        st.dataframe(
            tabela[[
                "CNES", "Município de atendimento", "Doença", "Faixa etária",
                "Internações", "Média observada (dias)", "Média prevista (dias)",
                "Observado − previsto (dias)",
            ]].round(2),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Baixar esta seleção em CSV",
            data=tabela.to_csv(index=False).encode("utf-8-sig"),
            file_name="comparacao_hospitais_filtrada.csv",
            mime="text/csv",
        )

with aba_deslocamentos:
    st.subheader("Município de residência e local de atendimento")
    if fluxo_saidas.empty:
        st.info("Execute o pipeline atualizado para gerar a contagem agregada de saídas da RMC.")
    if deslocamentos.empty:
        st.info(
            "Não encontrei TCC_Datasus/data/processed/gold_resp_eda.parquet. "
            "Execute o pipeline para gerar a base descritiva e atualizar esta análise."
        )
    else:
        deslocamentos = deslocamentos.copy()
        deslocamentos["ANO_CMPT"] = pd.to_numeric(
            deslocamentos["ANO_CMPT"], errors="coerce"
        )
        cod_res = deslocamentos["MUNIC_RES"].astype("string").str.replace(r"\.0$", "", regex=True).str[:6]
        cod_atend = deslocamentos["MUNIC_ATENDIMENTO"].astype("string").str.replace(r"\.0$", "", regex=True).str[:6]
        nome_res_norm = deslocamentos["NOME_MUNIC_RES"].map(normalizar_nome)
        deslocamentos["OUTRO_MUNICIPIO_RMC"] = (
            nome_res_norm.isin(MUNICIPIOS_RMC_NORMALIZADOS)
            & cod_res.notna() & cod_atend.notna() & (cod_res != cod_atend)
        )

        anos_conhecidos = set(deslocamentos["ANO_CMPT"].dropna().astype(int).unique())
        cidades_conhecidas = set(
            deslocamentos.loc[nome_res_norm.isin(MUNICIPIOS_RMC_NORMALIZADOS), "NOME_MUNIC_RES"]
            .dropna().astype(str).unique()
        )
        if not fluxo_saidas.empty:
            fluxo_saidas = fluxo_saidas.copy()
            fluxo_saidas["ANO_CMPT"] = pd.to_numeric(fluxo_saidas["ANO_CMPT"], errors="coerce")
            fluxo_saidas["INTERNACOES_FORA_RMC"] = pd.to_numeric(
                fluxo_saidas["INTERNACOES_FORA_RMC"], errors="coerce"
            ).fillna(0)
            anos_conhecidos.update(fluxo_saidas["ANO_CMPT"].dropna().astype(int).unique())
            cidades_conhecidas.update(fluxo_saidas["NOME_MUNIC_RES"].dropna().astype(str).unique())
        anos = sorted(anos_conhecidos)
        cidades_residencia = sorted(cidades_conhecidas)
        filtros_deslocamento = st.columns(2)
        with filtros_deslocamento[0]:
            anos_selecionados = st.multiselect(
                "Ano de atendimento", anos, default=anos, key="anos_deslocamento"
            )
        with filtros_deslocamento[1]:
            cidade_residencia = st.selectbox(
                "Município de residência",
                ["Todos"] + cidades_residencia,
                key="residencia_deslocamento",
            )

        dados_deslocamento = deslocamentos[
            deslocamentos["ANO_CMPT"].isin(anos_selecionados)
            & nome_res_norm.isin(MUNICIPIOS_RMC_NORMALIZADOS)
        ].copy()
        if fluxo_saidas.empty:
            dados_saidas = pd.DataFrame(
                columns=["ANO_CMPT", "NOME_MUNIC_RES", "INTERNACOES_FORA_RMC"]
            )
        else:
            dados_saidas = fluxo_saidas[fluxo_saidas["ANO_CMPT"].isin(anos_selecionados)].copy()
        if cidade_residencia != "Todos":
            dados_deslocamento = dados_deslocamento[
                dados_deslocamento["NOME_MUNIC_RES"].astype(str) == cidade_residencia
            ]
            dados_saidas = dados_saidas[
                dados_saidas["NOME_MUNIC_RES"].astype(str) == cidade_residencia
            ]

        if dados_deslocamento.empty and dados_saidas.empty:
            st.warning("Não há internações para os filtros selecionados.")
        else:
            interno_por_ano = (
                dados_deslocamento.groupby("ANO_CMPT")
                .agg(
                    total_interno=("ANO_CMPT", "size"),
                    outro_municipio_rmc=("OUTRO_MUNICIPIO_RMC", "sum"),
                )
                .reset_index()
            )
            saidas_por_ano = (
                dados_saidas.groupby("ANO_CMPT")
                .agg(
                    fora_rmc=("INTERNACOES_FORA_RMC", "sum"),
                )
                .reset_index()
            )
            por_ano = interno_por_ano.merge(saidas_por_ano, on="ANO_CMPT", how="outer").fillna(0)
            por_ano["local_conhecido"] = por_ano["total_interno"] + por_ano["fora_rmc"]
            por_ano["Fora da RMC"] = (
                100 * por_ano["fora_rmc"] / por_ano["local_conhecido"].replace(0, float("nan"))
            )
            por_ano["Outro município da RMC"] = (
                100 * por_ano["outro_municipio_rmc"] / por_ano["local_conhecido"].replace(0, float("nan"))
            )
            local_conhecido = int(por_ano["local_conhecido"].sum())
            outro_municipio_rmc = int(por_ano["outro_municipio_rmc"].sum())
            fora_rmc = int(por_ano["fora_rmc"].sum())
            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Atendidas em outra cidade da RMC",
                f"{outro_municipio_rmc / local_conhecido:.1%}" if local_conhecido else "—",
                f"{outro_municipio_rmc:,} internações".replace(",", "."),
            )
            k2.metric(
                "Internações atendidas fora da RMC",
                f"{fora_rmc / local_conhecido:.1%}" if local_conhecido else "—",
                f"{fora_rmc:,} internações".replace(",", "."),
            )
            k3.metric(
                "Atendimentos com município conhecido",
                f"{local_conhecido:,}".replace(",", "."),
            )

            serie = por_ano.melt(
                id_vars="ANO_CMPT",
                value_vars=["Outro município da RMC", "Fora da RMC"],
                var_name="Indicador",
                value_name="Internações (%)",
            )
            fig_deslocamentos = px.line(
                serie,
                x="ANO_CMPT",
                y="Internações (%)",
                color="Indicador",
                markers=True,
                title="Percentual de internações por local de atendimento",
                labels={"ANO_CMPT": "Ano"},
            )
            st.plotly_chart(fig_deslocamentos, use_container_width=True)

            st.caption(
                "Os municípios de atendimento externos à RMC aparecem somente na contagem agregada "
                "de saídas; os destinos não são discriminados. A série também mostra deslocamentos "
                "entre municípios da própria RMC. Os indicadores não explicam o motivo do deslocamento."
            )

with aba_perfil:
    st.subheader("Quais diagnósticos têm internações mais longas?")
    if deslocamentos.empty:
        st.info("A base descritiva não foi encontrada. Execute o pipeline para gerar os dados.")
    else:
        perfil = deslocamentos[
            deslocamentos["NOME_MUNIC_ATENDIMENTO"]
            .map(normalizar_nome)
            .isin(MUNICIPIOS_RMC_NORMALIZADOS)
        ].copy()
        perfil["ANO_CMPT"] = pd.to_numeric(perfil["ANO_CMPT"], errors="coerce")
        perfil["DIAS_PERM"] = pd.to_numeric(perfil["DIAS_PERM"], errors="coerce")
        perfil["MORTE"] = pd.to_numeric(perfil["MORTE"], errors="coerce")

        anos_perfil = sorted(perfil["ANO_CMPT"].dropna().astype(int).unique())
        opcoes_faixa = sorted(perfil["FAIXA_ETARIA"].dropna().astype(str).unique())
        opcoes_tipo = sorted(perfil["TIPO_ATENDIMENTO"].dropna().astype(str).unique())
        opcoes_grupo = sorted(perfil["SUBCATEGORIA_CID"].dropna().astype(str).unique())
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            anos_perfil_sel = st.multiselect(
                "Ano", anos_perfil, default=anos_perfil, key="anos_perfil"
            )
        with f2:
            faixa_perfil = st.selectbox(
                "Faixa etária", ["Todas"] + opcoes_faixa, key="faixa_perfil"
            )
        with f3:
            tipo_perfil = st.selectbox(
                "Tipo de atendimento", ["Todos"] + opcoes_tipo, key="tipo_perfil"
            )
        with f4:
            grupo_perfil = st.selectbox(
                "Grupo respiratório", ["Todos"] + opcoes_grupo, key="grupo_perfil"
            )

        saida_perfil = st.selectbox(
            "Situação na saída",
            ["Todos", "Sem óbito", "Com óbito"],
            key="saida_perfil",
        )
        dados_perfil = perfil[
            perfil["ANO_CMPT"].isin(anos_perfil_sel)
            & perfil["DIAS_PERM"].notna()
        ].copy()
        if faixa_perfil != "Todas":
            dados_perfil = dados_perfil[
                dados_perfil["FAIXA_ETARIA"].astype(str) == faixa_perfil
            ]
        if tipo_perfil != "Todos":
            dados_perfil = dados_perfil[
                dados_perfil["TIPO_ATENDIMENTO"].astype(str) == tipo_perfil
            ]
        if grupo_perfil != "Todos":
            dados_perfil = dados_perfil[
                dados_perfil["SUBCATEGORIA_CID"].astype(str) == grupo_perfil
            ]
        if saida_perfil == "Sem óbito":
            dados_perfil = dados_perfil[dados_perfil["MORTE"] != 1]
        elif saida_perfil == "Com óbito":
            dados_perfil = dados_perfil[dados_perfil["MORTE"] == 1]

        resumo_perfil = (
            dados_perfil.groupby(
                ["SUBCATEGORIA_CID", "DIAG_PRINC"],
                dropna=False,
            )
            .agg(
                internacoes=("DIAS_PERM", "size"),
                media=("DIAS_PERM", "mean"),
                mediana=("DIAS_PERM", "median"),
                p75=("DIAS_PERM", lambda valores: valores.quantile(0.75)),
                p90=("DIAS_PERM", lambda valores: valores.quantile(0.90)),
            )
            .reset_index()
        )
        resumo_perfil = resumo_perfil[resumo_perfil["internacoes"] >= 30].copy()
        if resumo_perfil.empty:
            st.warning(
                "Nenhum diagnóstico teve 30 ou mais internações com esses filtros."
            )
        else:
            resumo_perfil = resumo_perfil.sort_values(
                ["p90", "internacoes"], ascending=[False, False]
            )
            fig_perfil = px.bar(
                resumo_perfil.head(15).sort_values("p90"),
                x="p90",
                y="DIAG_PRINC",
                color="SUBCATEGORIA_CID",
                orientation="h",
                hover_data=["internacoes", "mediana", "media"],
                labels={
                    "p90": "P90 da permanência (dias)",
                    "DIAG_PRINC": "CID-10 principal",
                    "SUBCATEGORIA_CID": "Grupo respiratório",
                    "internacoes": "Internações",
                },
                title="Diagnósticos com maiores P90 de permanência (mínimo 30 casos)",
            )
            st.plotly_chart(fig_perfil, use_container_width=True)
            tabela_perfil = resumo_perfil.rename(
                columns={
                    "SUBCATEGORIA_CID": "Grupo respiratório",
                    "DIAG_PRINC": "CID-10 principal",
                    "internacoes": "Internações",
                    "media": "Média (dias)",
                    "mediana": "Mediana (dias)",
                    "p75": "P75 (dias)",
                    "p90": "P90 (dias)",
                }
            )
            st.dataframe(
                tabela_perfil.round(2),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "P90 é o valor abaixo do qual ficam 90% das permanências do grupo; ajuda a "
                "enxergar internações longas que a média pode esconder. A análise é descritiva, "
                "inclui filtro para óbitos e não mede qualidade hospitalar."
            )

with aba_previsao_grupos:
    st.subheader("Em quais grupos a previsão ajuda?")
    if metricas_por_grupo.empty:
        st.info(
            "Ainda não encontrei metricas_por_grupo_2025.csv. Salve a alteração e execute "
            "analise_permanencia_rmc.py para gerar as métricas por grupo."
        )
    else:
        grupos = metricas_por_grupo.copy()
        for coluna in [
            "internacoes", "mae_baseline", "mae_sem_municipio",
            "mae_com_municipio", "ganho_mae_sem_municipio",
            "ganho_mae_com_municipio", "mae_hgbr", "ganho_mae_hgbr",
        ]:
            grupos[coluna] = pd.to_numeric(grupos[coluna], errors="coerce")
        filtros_grupo = st.columns(3)
        categorias_grupo = sorted(grupos["SUBCATEGORIA_CID"].dropna().astype(str).unique())
        faixas_grupo = sorted(grupos["FAIXA_ETARIA"].dropna().astype(str).unique())
        tipos_grupo = sorted(grupos["TIPO_ATENDIMENTO"].dropna().astype(str).unique())
        with filtros_grupo[0]:
            categoria_sel = st.selectbox(
                "Doença", ["Todas"] + categorias_grupo, key="categoria_erro"
            )
        with filtros_grupo[1]:
            faixa_sel = st.selectbox(
                "Faixa etária", ["Todas"] + faixas_grupo, key="faixa_erro"
            )
        with filtros_grupo[2]:
            tipo_sel = st.selectbox(
                "Tipo de atendimento", ["Todos"] + tipos_grupo, key="tipo_erro"
            )
        if categoria_sel != "Todas":
            grupos = grupos[grupos["SUBCATEGORIA_CID"].astype(str) == categoria_sel]
        if faixa_sel != "Todas":
            grupos = grupos[grupos["FAIXA_ETARIA"].astype(str) == faixa_sel]
        if tipo_sel != "Todos":
            grupos = grupos[grupos["TIPO_ATENDIMENTO"].astype(str) == tipo_sel]

        if grupos.empty:
            st.warning("Não há grupos com ao menos 30 internações para esses filtros.")
        else:
            linha_modelos = grupos.melt(
                id_vars=[
                    "SUBCATEGORIA_CID", "FAIXA_ETARIA",
                    "TIPO_ATENDIMENTO", "internacoes",
                ],
                value_vars=[
                    "mae_baseline", "mae_sem_municipio", "mae_com_municipio",
                    "mae_hgbr",
                ],
                var_name="Modelo",
                value_name="MAE (dias)",
            )
            nomes_modelos = {
                "mae_baseline": "Média de referência",
                "mae_sem_municipio": "Modelo sem município",
                "mae_com_municipio": "Modelo com município",
                "mae_hgbr": "HistGradientBoosting",
            }
            linha_modelos["Modelo"] = linha_modelos["Modelo"].map(nomes_modelos)
            linha_modelos["Grupo"] = (
                linha_modelos["SUBCATEGORIA_CID"].astype(str)
                + " | "
                + linha_modelos["FAIXA_ETARIA"].astype(str)
                + " | "
                + linha_modelos["TIPO_ATENDIMENTO"].astype(str)
            )
            fig_erros = px.bar(
                linha_modelos,
                x="Grupo",
                y="MAE (dias)",
                color="Modelo",
                barmode="group",
                hover_data=["internacoes"],
                title="Erro médio absoluto por grupo no teste de 2025",
            )
            st.plotly_chart(fig_erros, use_container_width=True)

            tabela_grupos = grupos.rename(
                columns={
                    "SUBCATEGORIA_CID": "Doença",
                    "FAIXA_ETARIA": "Faixa etária",
                    "TIPO_ATENDIMENTO": "Tipo de atendimento",
                    "internacoes": "Internações",
                    "mediana_observada": "Mediana observada (dias)",
                    "p90_observado": "P90 observado (dias)",
                    "mae_baseline": "MAE da média (dias)",
                    "mae_sem_municipio": "MAE modelo sem município (dias)",
                    "mae_com_municipio": "MAE modelo com município (dias)",
                    "mae_hgbr": "MAE HistGradientBoosting (dias)",
                    "ganho_mae_sem_municipio": "Redução do erro sem município (dias)",
                    "ganho_mae_com_municipio": "Redução do erro com município (dias)",
                    "ganho_mae_hgbr": "Redução do erro HistGradientBoosting (dias)",
                }
            )
            st.dataframe(
                tabela_grupos.round(2),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Redução positiva significa que o modelo teve MAE menor que a previsão pela média; "
                "valor negativo significa que errou mais. Os grupos têm pelo menos 30 internações. "
                "O HistGradientBoosting usa as variáveis do modelo original (idade, faixa etária, "
                "estação, grupo CID e tipo de atendimento); os modelos Random Forest também incluem "
                "sexo e, na versão indicada, município. A comparação é entre essas versões completas, "
                "com previsões para 2025, e não prova causalidade."
            )

with aba_metodo:
    st.subheader("O que esta comparação mostra")
    st.markdown(
        "- **Observado − previsto positivo:** o grupo teve média de permanência acima da previsão.\n"
        "- **Observado − previsto negativo:** o grupo teve média abaixo da previsão.\n"
        "- **Próximo de zero:** observado e previsto ficaram próximos para aquele grupo."
    )
    st.warning(
        "A diferença não mede diretamente qualidade nem permite declarar um hospital como o melhor. "
        "O modelo usa apenas as características disponíveis no conjunto de dados e pode não capturar "
        "gravidade, comorbidades, transferências ou outros fatores clínicos."
    )
    st.markdown(
        "Os estabelecimentos são identificados pelo **CNES**. O município exibido é o município de "
        "atendimento. Para apresentar nomes oficiais e históricos dos estabelecimentos, a etapa seguinte "
        "é cruzar os códigos CNES com o cadastro referente à competência de 2025."
    )

st.divider()
st.caption(
    "Fonte: arquivos agregados do projeto. O painel não exibe registros individuais de pacientes."
)
