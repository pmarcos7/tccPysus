import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Configuração da página
st.set_page_config(page_title="Dashboard Evasão SUS", layout="wide")

# Dicionário de Coordenadas da RMC
COORDENADAS_RMC = {
    'CAMPINAS': (-22.9099, -47.0626),
    'AMERICANA': (-22.7392, -47.3315),
    'ARTUR NOGUEIRA': (-22.5732, -47.1728),
    'COSMOPOLIS': (-22.6461, -47.1996),
    'ENGENHEIRO COELHO': (-22.4868, -47.2023),
    'HOLAMBRA': (-22.6334, -47.0569),
    'HORTOLANDIA': (-22.8583, -47.2200),
    'INDAIATUBA': (-23.0903, -47.2181),
    'ITATIBA': (-23.0016, -46.8354),
    'JAGUARIUNA': (-22.7051, -46.9856),
    'MONTE MOR': (-22.9495, -47.3146),
    'MORUNGABA': (-22.8837, -46.7909),
    'NOVA ODESSA': (-22.7788, -47.2954),
    'PAULINIA': (-22.7618, -47.1534),
    'PEDREIRA': (-22.7423, -46.9015),
    "SANTA BARBARA D'OESTE": (-22.7554, -47.4146),
    'SANTO ANTONIO DE POSSE': (-22.6074, -46.9176),
    'SUMARE': (-22.8206, -47.2667),
    'VALINHOS': (-22.9705, -46.9956),
    'VINHEDO': (-23.0298, -46.9749),
    'NAO INFORMADO': (None, None)
}

# 1. Carregar os Dados
@st.cache_data
def carregar_dados():
    caminho = './TCC_Datasus/data/processed/gold_resp_eda.parquet'
    if not os.path.exists(caminho):
        st.error(f"Ficheiro não encontrado: {caminho}")
        return pd.DataFrame()
    return pd.read_parquet(caminho)

df = carregar_dados()

if not df.empty:
    st.title("🗺️ Mapa de Fluxo de Evasão Hospitalar (Doenças Respiratórias)")
    st.markdown("Linhas representam pacientes atendidos fora do município de residência na RMC.")

    # Normalizar nomes para fazer o 'match' com as coordenadas
    import unicodedata
    def normalizar(s):
        s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
        return s.upper().strip()

    df['MUN_RES_NORM'] = df['NOME_MUNIC_RES'].apply(normalizar)
    df['MUN_ATEND_NORM'] = df['NOME_MUNIC_ATENDIMENTO'].apply(normalizar)

    # 2. Filtrar apenas casos de evasão (ATENDIDO_FORA == 1)
    df_evasao = df[df['ATENDIDO_FORA'] == 1].copy()

    # 3. Agrupar volume de evasão (Origem -> Destino)
    fluxo = df_evasao.groupby(['MUN_RES_NORM', 'MUN_ATEND_NORM']).size().reset_index(name='TOTAL')
    
    # 4. Construir o Mapa
    # 4. Construir o Mapa (Código Atualizado para Plotly >= 5.24)
    fig = go.Figure()

    # Adicionar as linhas de fluxo
    for _, row in fluxo.iterrows():
        origem = row['MUN_RES_NORM']
        destino = row['MUN_ATEND_NORM']
        total = row['TOTAL']
        
        coord_origem = COORDENADAS_RMC.get(origem)
        coord_destino = COORDENADAS_RMC.get(destino)
        
        if coord_origem and coord_destino and coord_origem[0] and coord_destino[0]:
            # Adiciona a linha usando a nova nomenclatura Scattermap
            fig.add_trace(go.Scattermap(
                mode="lines",
                lon=[coord_origem[1], coord_destino[1]],
                lat=[coord_origem[0], coord_destino[0]],
                line=dict(width=total / 20, color='red'),  # Espessura baseada no volume
                opacity=0.6,
                name=f"{origem} -> {destino} ({total})",
                hoverinfo="name"
            ))

    # Adicionar marcadores para as cidades
    lats = [c[0] for c in COORDENADAS_RMC.values() if c[0] is not None]
    lons = [c[1] for c in COORDENADAS_RMC.values() if c[1] is not None]
    nomes = [nome for nome, c in COORDENADAS_RMC.items() if c[0] is not None]

    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lon=lons,
        lat=lats,
        marker=dict(size=10, color='royalblue'),
        text=nomes,
        textposition="bottom right",
        hoverinfo="text"
    ))

    # Configuração do Layout do Mapa usando 'map' em vez de 'mapbox'
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=-22.9099, lon=-47.0626), # Centrado em Campinas
            zoom=9
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabela resumo abaixo do mapa
    st.subheader("Top 10 Rotas de Evasão")
    st.dataframe(fluxo.sort_values('TOTAL', ascending=False).head(10))