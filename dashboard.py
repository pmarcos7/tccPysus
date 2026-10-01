import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import unicodedata

st.set_page_config(page_title="Dashboard Evasão SUS", layout="wide")

COORDENADAS_RMC = {
    'CAMPINAS': (-22.9099, -47.0626), 'AMERICANA': (-22.7392, -47.3315),
    'ARTUR NOGUEIRA': (-22.5732, -47.1728), 'COSMOPOLIS': (-22.6461, -47.1996),
    'ENGENHEIRO COELHO': (-22.4868, -47.2023), 'HOLAMBRA': (-22.6334, -47.0569),
    'HORTOLANDIA': (-22.8583, -47.2200), 'INDAIATUBA': (-23.0903, -47.2181),
    'ITATIBA': (-23.0016, -46.8354), 'JAGUARIUNA': (-22.7051, -46.9856),
    'MONTE MOR': (-22.9495, -47.3146), 'MORUNGABA': (-22.8837, -46.7909),
    'NOVA ODESSA': (-22.7788, -47.2954), 'PAULINIA': (-22.7618, -47.1534),
    'PEDREIRA': (-22.7423, -46.9015), "SANTA BARBARA D'OESTE": (-22.7554, -47.4146),
    'SANTO ANTONIO DE POSSE': (-22.6074, -46.9176), 'SUMARE': (-22.8206, -47.2667),
    'VALINHOS': (-22.9705, -46.9956), 'VINHEDO': (-23.0298, -46.9749)
}

@st.cache_data
def carregar_dados():
    caminho = './TCC_Datasus/data/processed/gold_resp_eda.parquet'
    if not os.path.exists(caminho):
        st.error(f"Ficheiro não encontrado: {caminho}")
        return pd.DataFrame()
    return pd.read_parquet(caminho)

def normalizar(s):
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii').upper().strip()

df = carregar_dados()

if not df.empty:
    st.title("🗺️ Mapa de Fluxo de Evasão Hospitalar na RMC")
    
    # Controlos de Interface para Otimização de Performance
    col1, col2 = st.columns(2)
    with col1:
        min_pacientes = st.slider("Esconder rotas com menos de X pacientes (Otimiza a velocidade)", min_value=1, max_value=500, value=20)
    with col2:
        foco_direcao = st.radio("Isolar Direção do Fluxo:", ["Ver Toda a Rede", "Destino: Internados EM Campinas", "Origem: Saíram DE Campinas"])

    df['MUN_RES_NORM'] = df['NOME_MUNIC_RES'].apply(normalizar)
    df['MUN_ATEND_NORM'] = df['NOME_MUNIC_ATENDIMENTO'].apply(normalizar)

    df_evasao = df[df['ATENDIDO_FORA'] == 1].copy()
    fluxo = df_evasao.groupby(['MUN_RES_NORM', 'MUN_ATEND_NORM']).size().reset_index(name='TOTAL')
    
    # Aplicar Filtros da Interface
    fluxo = fluxo[fluxo['TOTAL'] >= min_pacientes]
    
    if foco_direcao == "Destino: Internados EM Campinas":
        fluxo = fluxo[fluxo['MUN_ATEND_NORM'] == 'CAMPINAS']
    elif foco_direcao == "Origem: Saíram DE Campinas":
        fluxo = fluxo[fluxo['MUN_RES_NORM'] == 'CAMPINAS']

    fig = go.Figure()

    mid_lats, mid_lons, mid_texts, mid_values = [], [], [], []

    for _, row in fluxo.iterrows():
        origem = row['MUN_RES_NORM']
        destino = row['MUN_ATEND_NORM']
        total = row['TOTAL']
        
        coord_origem = COORDENADAS_RMC.get(origem)
        coord_destino = COORDENADAS_RMC.get(destino)
        
        if coord_origem and coord_destino:
            # Lógica de Cores por Direção
            if destino == 'CAMPINAS':
                cor_linha = '#d62728' # Vermelho (Entrada no Polo)
            elif origem == 'CAMPINAS':
                cor_linha = '#1f77b4' # Azul (Saída do Polo)
            else:
                cor_linha = '#7f7f7f' # Cinzento (Fluxo Periférico)

            # Desenha a linha da rota
            fig.add_trace(go.Scattermap(
                mode="lines",
                lon=[coord_origem[1], coord_destino[1]],
                lat=[coord_origem[0], coord_destino[0]],
                line=dict(width=max(1, total / 50), color=cor_linha),
                opacity=0.6,
                name=f"{origem} ➔ {destino}",
                hoverinfo="name"
            ))

            # Guarda os pontos centrais para colocar o número depois
            mid_lats.append((coord_origem[0] + coord_destino[0]) / 2)
            mid_lons.append((coord_origem[1] + coord_destino[1]) / 2)
            mid_values.append(str(total))
            mid_texts.append(f"<b>{total} pacientes</b><br>{origem} ➔ {destino}")

    # Adicionar uma única camada com todos os números nos pontos centrais (Altamente otimizado)
    if mid_lats:
        fig.add_trace(go.Scattermap(
            mode="text",
            lon=mid_lons,
            lat=mid_lats,
            text=mid_values,
            textfont=dict(size=14, color='black', family="Arial Black"),
            hovertext=mid_texts,
            hoverinfo="text"
        ))

    # Adicionar os marcadores das cidades
    lats = [c[0] for c in COORDENADAS_RMC.values() if c[0] is not None]
    lons = [c[1] for c in COORDENADAS_RMC.values() if c[1] is not None]
    nomes = [nome for nome, c in COORDENADAS_RMC.items() if c[0] is not None]

    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lon=lons,
        lat=lats,
        marker=dict(size=12, color='#2ca02c', opacity=0.8),
        text=nomes,
        textposition="bottom right",
        textfont=dict(size=11, color='white'),
        hoverinfo="text"
    ))

    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=-22.9099, lon=-47.0626),
            zoom=9.5
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        height=750
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Top Rotas de Evasão (Mínimo de {min_pacientes} pacientes)")
    st.dataframe(fluxo.sort_values('TOTAL', ascending=False).reset_index(drop=True), use_container_width=True)


# (Coloque isto no seu dashboard.py, abaixo do mapa de evasão)

st.divider()
st.title("🔥 Mapa de Calor: Focos de Doenças Respiratórias em Campinas")

# 1. Carregar as coordenadas cacheadas
@st.cache_data
def carregar_coords_bairros():
    caminho = './TCC_Datasus/data/processed/coords_bairros_campinas.csv'
    if os.path.exists(caminho):
        return pd.read_csv(caminho)
    return pd.DataFrame()

df_coords = carregar_coords_bairros()

if not df_coords.empty and not df.empty:
    # 2. Filtrar apenas pacientes residentes em Campinas
    df_campinas = df[df['MUNIC_RES'].astype(str).str.startswith('350950')].copy()
    
    # 3. Agrupar volume por Bairro
    focos = df_campinas.groupby('NOME_BAIRRO').size().reset_index(name='TOTAL_INTERNACOES')
    
    # 4. Juntar o volume com as latitudes e longitudes
    heatmap_data = focos.merge(df_coords, left_on='NOME_BAIRRO', right_on='BAIRRO', how='inner')
    # Remover bairros que a API não encontrou
    heatmap_data = heatmap_data.dropna(subset=['LAT', 'LON'])

    # 5. Criar o Mapa de Calor
    fig_heat = go.Figure(go.Densitymap(
        lat=heatmap_data['LAT'],
        lon=heatmap_data['LON'],
        z=heatmap_data['TOTAL_INTERNACOES'], # O 'z' define a intensidade do calor (volume)
        radius=25, # Tamanho do raio de dissipação do calor
        colorscale='Inferno',
        name="Internações",
        hoverinfo="text",
        text=heatmap_data['NOME_BAIRRO'] + ": " + heatmap_data['TOTAL_INTERNACOES'].astype(str) + " casos"
    ))

    fig_heat.update_layout(
        map=dict(
            style="carto-darkmatter", # Fundo escuro destaca melhor mapas de calor
            center=dict(lat=-22.9099, lon=-47.0626), # Centro de Campinas
            zoom=11
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600
    )

    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Execute o script de geocodificação para gerar o CSV de coordenadas dos bairros de Campinas primeiro.")
