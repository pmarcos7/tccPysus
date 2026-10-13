import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
from utils import MUNICIPIOS_RMC_NORMALIZADOS, normalizar_nome

# =========================================================
# CONFIGURAÇÕES INICIAIS
# =========================================================
st.set_page_config(page_title="Dashboard SUS - RMC", layout="wide")

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

def classificar_cid10(cid):
    if pd.isna(cid): return "Não Informado"
    prefixo = str(cid).upper()[:3]
    if 'J00' <= prefixo <= 'J06': return "Infecções Agudas (Vias Superiores)"
    if 'J09' <= prefixo <= 'J18': return "Influenza e Pneumonias"
    if 'J20' <= prefixo <= 'J22': return "Infecções Agudas (Vias Inferiores)"
    if 'J30' <= prefixo <= 'J39': return "Rinite, Sinusite e Vias Sup."
    if 'J40' <= prefixo <= 'J47': return "Crônicas (Asma, DPOC, Bronquite)"
    if 'J60' <= prefixo <= 'J70': return "Doenças Pulmonares Externas"
    return "Outras Doenças Respiratórias"

@st.cache_data
def carregar_dados():
    caminho = './TCC_Datasus/data/processed/gold_resp_eda.parquet'
    if not os.path.exists(caminho):
        st.error(f"Ficheiro não encontrado: {caminho}")
        return pd.DataFrame()
    return pd.read_parquet(caminho)

@st.cache_data
def carregar_fluxo_saidas():
    caminho = './TCC_Datasus/data/processed/gold_resp_fluxo.parquet'
    if not os.path.exists(caminho):
        return pd.DataFrame()
    return pd.read_parquet(caminho)

@st.cache_data
def carregar_coords_bairros():
    caminho = './TCC_Datasus/data/processed/coords_bairros_campinas.csv'
    if os.path.exists(caminho):
        return pd.read_csv(caminho)
    return pd.DataFrame()

df = carregar_dados()
fluxo_saidas = carregar_fluxo_saidas()

if not df.empty:
    df['MUN_RES_NORM'] = df['NOME_MUNIC_RES'].apply(normalizar_nome)
    df['MUN_ATEND_NORM'] = df['NOME_MUNIC_ATENDIMENTO'].apply(normalizar_nome)
    atendimento_na_rmc = df['MUN_ATEND_NORM'].isin(MUNICIPIOS_RMC_NORMALIZADOS)
    df_rmc_atendimento = df[atendimento_na_rmc].copy()
    total_saidas = (
        int(pd.to_numeric(fluxo_saidas.get('INTERNACOES_FORA_RMC', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())
        if not fluxo_saidas.empty else 0
    )
    
    aba_evasao, aba_incidencia = st.tabs(["🚑 Fluxo e Evasão Hospitalar", "🔥 Monitoramento Epidemiológico e Calor"])

    # =========================================================
    # ABA 1: FLUXO DE EVASÃO
    # =========================================================
    with aba_evasao:
        st.title("🗺️ Deslocamentos para atendimento na RMC")
        if fluxo_saidas.empty:
            st.info("Execute o pipeline atualizado para gerar a contagem agregada de saídas da RMC.")

        total_com_local = len(df_rmc_atendimento) + total_saidas
        percentual_saidas = total_saidas / total_com_local if total_com_local else 0
        k_saida, k_percentual = st.columns(2)
        k_saida.metric("Internações de moradores da RMC atendidas fora", f"{total_saidas:,}".replace(',', '.'))
        k_percentual.metric("Percentual entre atendimentos com local conhecido", f"{percentual_saidas:.1%}")
        st.caption("Os destinos externos são contabilizados em conjunto; o dashboard não informa em quais municípios ocorreram.")
        
        col1, col2 = st.columns(2)
        with col1:
            min_pacientes = st.slider("Esconder rotas com menos de X pacientes:", min_value=1, max_value=500, value=20)
        with col2:
            foco_direcao = st.radio("Filtrar fluxos internos da RMC:", ["Ver Toda a Rede", "Destino: atendimento em Campinas", "Origem: saídas de Campinas para outra cidade da RMC"])

        # O mapa e as rotas mostram somente deslocamentos entre municípios da RMC.
        df_evasao = df_rmc_atendimento[df_rmc_atendimento['ATENDIDO_FORA'] == 1].copy()
        fluxo = df_evasao.groupby(['MUN_RES_NORM', 'MUN_ATEND_NORM']).size().reset_index(name='TOTAL')
        
        fluxo_mapa = fluxo[fluxo['TOTAL'] >= min_pacientes]
        if foco_direcao == "Destino: atendimento em Campinas":
            fluxo_mapa = fluxo_mapa[fluxo_mapa['MUN_ATEND_NORM'] == 'CAMPINAS']
        elif foco_direcao == "Origem: saídas de Campinas para outra cidade da RMC":
            fluxo_mapa = fluxo_mapa[fluxo_mapa['MUN_RES_NORM'] == 'CAMPINAS']

        fig_fluxo = go.Figure()
        mid_lats, mid_lons, mid_texts, mid_values = [], [], [], []

        for _, row in fluxo_mapa.iterrows():
            origem = row['MUN_RES_NORM']
            destino = row['MUN_ATEND_NORM']
            total = row['TOTAL']
            coord_origem = COORDENADAS_RMC.get(origem)
            coord_destino = COORDENADAS_RMC.get(destino)
            
            if coord_origem and coord_destino:
                if destino == 'CAMPINAS': cor_linha = '#d62728' 
                elif origem == 'CAMPINAS': cor_linha = '#1f77b4' 
                else: cor_linha = '#7f7f7f' 

                fig_fluxo.add_trace(go.Scattermap(
                    mode="lines", lon=[coord_origem[1], coord_destino[1]], lat=[coord_origem[0], coord_destino[0]],
                    line=dict(width=max(1, total / 50), color=cor_linha), opacity=0.6,
                    name=f"{origem} ➔ {destino}", hoverinfo="name"
                ))

                mid_lats.append((coord_origem[0] + coord_destino[0]) / 2)
                mid_lons.append((coord_origem[1] + coord_destino[1]) / 2)
                mid_values.append(str(total))
                mid_texts.append(f"<b>{total} pacientes</b><br>{origem} ➔ {destino}")

        if mid_lats:
            fig_fluxo.add_trace(go.Scattermap(
                mode="text", lon=mid_lons, lat=mid_lats, text=mid_values,
                textfont=dict(size=14, color='black', family="Arial Black"),
                hovertext=mid_texts, hoverinfo="text"
            ))

        lats = [c[0] for c in COORDENADAS_RMC.values() if c[0] is not None]
        lons = [c[1] for c in COORDENADAS_RMC.values() if c[1] is not None]
        nomes = [nome for nome, c in COORDENADAS_RMC.items() if c[0] is not None]

        fig_fluxo.add_trace(go.Scattermap(
            mode="markers+text", lon=lons, lat=lats, marker=dict(size=12, color='#2ca02c', opacity=0.8),
            text=nomes, textposition="bottom right", textfont=dict(size=11, color='white'), hoverinfo="text"
        ))

        fig_fluxo.update_layout(
            map=dict(style="carto-positron", center=dict(lat=-22.9099, lon=-47.0626), zoom=9.5),
            margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False, height=650
        )

        st.plotly_chart(fig_fluxo, use_container_width=True)

        st.subheader("📋 Principais deslocamentos entre municípios da RMC")
        tabela_rotas = fluxo.sort_values('TOTAL', ascending=False).reset_index(drop=True)
        tabela_rotas.index = [f"{i+1}º" for i in range(len(tabela_rotas))]
        tabela_rotas = tabela_rotas.rename(columns={'MUN_RES_NORM': 'Cidade de Origem (Saiu de)', 'MUN_ATEND_NORM': 'Cidade de Destino (Foi para)', 'TOTAL': 'Total de Pacientes'})
        st.dataframe(tabela_rotas.head(15), use_container_width=True)

        st.divider()
        st.subheader("📊 Ranking de Cidades")
        
        c_origem, c_destino = st.columns(2)
        with c_origem:
            top_origem = df_evasao.groupby('MUN_RES_NORM').size().reset_index(name='TOTAL').sort_values('TOTAL', ascending=False).head(10)
            tabela_origem = top_origem.copy().reset_index(drop=True)
            tabela_origem.index = [f"{i+1}º" for i in range(len(tabela_origem))]
            tabela_origem = tabela_origem.rename(columns={'MUN_RES_NORM': 'Cidade Emissora', 'TOTAL': 'Total que Saíram'})
            st.dataframe(tabela_origem, use_container_width=True)
            
        with c_destino:
            top_destino = df_evasao.groupby('MUN_ATEND_NORM').size().reset_index(name='TOTAL').sort_values('TOTAL', ascending=False).head(10)
            tabela_destino = top_destino.copy().reset_index(drop=True)
            tabela_destino.index = [f"{i+1}º" for i in range(len(tabela_destino))]
            tabela_destino = tabela_destino.rename(columns={'MUN_ATEND_NORM': 'Cidade Receptora', 'TOTAL': 'Total Recebidos'})
            st.dataframe(tabela_destino, use_container_width=True)

        # --- Evolução dos deslocamentos dentro e para fora da RMC ---
        st.divider()
        st.subheader("📈 Deslocamentos dentro da RMC e saídas da região")
        
        evol_total_evasao = df_evasao.groupby('ANO_CMPT').size().reset_index(name='TOTAL')
        fig_evol_evasao = px.line(evol_total_evasao, x='ANO_CMPT', y='TOTAL', markers=True, text='TOTAL',
                                  labels={'ANO_CMPT': 'Ano', 'TOTAL': 'Deslocamentos entre municípios da RMC'})
        
        fig_evol_evasao.update_traces(textposition="top center", line_color='#ff4b4b', line_width=4, marker_size=10, textfont_size=14)
        fig_evol_evasao.update_layout(template="plotly_dark", xaxis=dict(type='category'), margin={"r":0,"t":10,"l":0,"b":0}, height=450)
        
        st.plotly_chart(fig_evol_evasao, use_container_width=True)

        saidas_por_ano = (
            fluxo_saidas.groupby('ANO_CMPT')['INTERNACOES_FORA_RMC'].sum()
            .rename('Moradores atendidos fora da RMC').reset_index()
            if not fluxo_saidas.empty else pd.DataFrame(columns=['ANO_CMPT', 'Moradores atendidos fora da RMC'])
        )
        fig_saidas = px.bar(
            saidas_por_ano,
            x='ANO_CMPT',
            y='Moradores atendidos fora da RMC',
            labels={'ANO_CMPT': 'Ano'},
            title='Internações de moradores da RMC atendidas fora — sem detalhar o destino',
        )
        st.plotly_chart(fig_saidas, use_container_width=True)


    # =========================================================
    # ABA 2: MAPA DE CALOR E INCIDÊNCIA
    # =========================================================
    with aba_incidencia:
        st.title("🔥 Internações respiratórias por bairro de residência em Campinas")
        
        df_coords = carregar_coords_bairros()

        if not df_coords.empty:
            df_campinas = df_rmc_atendimento[
                df_rmc_atendimento['MUNIC_RES'].astype(str).str.startswith('350950')
            ].copy()
            coluna_cid = 'DIAG_PRINC' if 'DIAG_PRINC' in df_campinas.columns else 'CID_PRI' if 'CID_PRI' in df_campinas.columns else None
            
            if coluna_cid: df_campinas['CATEGORIA_DOENCA'] = df_campinas[coluna_cid].apply(classificar_cid10)
            else: df_campinas['CATEGORIA_DOENCA'] = "Coluna de CID Não Encontrada"

            col_ano, col_estacao, col_doenca = st.columns(3)
            with col_ano:
                anos = ["Todos os Anos"] + sorted(df_campinas['ANO_CMPT'].unique().tolist())
                ano_selecionado = st.selectbox("Filtrar por Ano (Mapa):", anos)
            with col_estacao:
                if 'ESTACAO' in df_campinas.columns:
                    estacoes = ["Todas as Estações"] + sorted(df_campinas['ESTACAO'].dropna().unique().tolist())
                    estacao_selecionada = st.selectbox("Filtrar por Estação Climática:", estacoes)
                else: estacao_selecionada = "Todas as Estações"
            with col_doenca:
                doencas = ["Todas as Doenças"] + sorted(df_campinas['CATEGORIA_DOENCA'].unique().tolist())
                doenca_selecionada = st.selectbox("Filtrar por Classificação (CID-10):", doencas)

            df_comparativo = df_campinas.copy()
            
            if ano_selecionado != "Todos os Anos": df_campinas = df_campinas[df_campinas['ANO_CMPT'] == ano_selecionado]
            if estacao_selecionada != "Todas as Estações" and 'ESTACAO' in df_campinas.columns:
                df_campinas = df_campinas[df_campinas['ESTACAO'] == estacao_selecionada]
                df_comparativo = df_comparativo[df_comparativo['ESTACAO'] == estacao_selecionada] 
            if doenca_selecionada != "Todas as Doenças":
                df_campinas = df_campinas[df_campinas['CATEGORIA_DOENCA'] == doenca_selecionada]
                df_comparativo = df_comparativo[df_comparativo['CATEGORIA_DOENCA'] == doenca_selecionada] 

            if df_campinas.empty:
                st.warning("⚠️ Não existem internamentos para esta combinação de filtros.")
            else:
                focos = df_campinas.groupby('NOME_BAIRRO').size().reset_index(name='TOTAL_INTERNACOES')
                heatmap_data = focos.merge(df_coords, left_on='NOME_BAIRRO', right_on='BAIRRO', how='inner').dropna(subset=['LAT', 'LON'])
                lats_dispersos, lons_dispersos = [], []
                dispersao = 0.012 

                for _, row in heatmap_data.iterrows():
                    lats_dispersos.extend(np.random.normal(loc=row['LAT'], scale=dispersao, size=row['TOTAL_INTERNACOES']))
                    lons_dispersos.extend(np.random.normal(loc=row['LON'], scale=dispersao, size=row['TOTAL_INTERNACOES']))

                fig_heat = go.Figure(go.Densitymap(lat=lats_dispersos, lon=lons_dispersos, radius=5, colorscale='Inferno', opacity=0.5))
                fig_heat.update_layout(map=dict(style="carto-darkmatter", center=dict(lat=-22.9099, lon=-47.0626), zoom=11), margin={"r":0,"t":0,"l":0,"b":0}, height=650)
                st.plotly_chart(fig_heat, use_container_width=True)

            st.subheader(f"📋 Top Bairros de Campinas com mais Internações")
            if not df_campinas.empty:
                tabela_bairros = focos.sort_values('TOTAL_INTERNACOES', ascending=False).reset_index(drop=True).head(10)
                tabela_bairros.index = [f"{i+1}º" for i in range(len(tabela_bairros))]
                tabela_bairros = tabela_bairros.rename(columns={'NOME_BAIRRO': 'Bairro de Residência', 'TOTAL_INTERNACOES': 'Total Identificado'})
                st.dataframe(tabela_bairros, use_container_width=True)

            # --- GRÁFICOS DE LINHAS (ABA 2) ---
            st.divider()
            
            if not df_comparativo.empty:
                if doenca_selecionada != "Todas as Doenças":
                    # Mostra apenas um gráfico central se filtrou apenas uma doença
                    st.subheader(f"📈 Evolução de Casos: {doenca_selecionada}")
                    evolucao = df_comparativo.groupby('ANO_CMPT').size().reset_index(name='TOTAL')
                    
                    fig_evolucao = px.line(
                        evolucao, x='ANO_CMPT', y='TOTAL', markers=True, text='TOTAL',
                        labels={'ANO_CMPT': 'Ano', 'TOTAL': 'Nº de Internações'}
                    )
                    fig_evolucao.update_traces(textposition="top center", line_color='#ff4b4b', line_width=4, marker_size=10, textfont_size=14)
                    fig_evolucao.update_layout(template="plotly_dark", margin={"r":0,"t":10,"l":0,"b":0}, xaxis=dict(type='category'), height=450)
                    
                    st.plotly_chart(fig_evolucao, use_container_width=True)
                
                else:
                    # Mostra os dois gráficos lado a lado se estiver em "Todas as Doenças"
                    st.subheader("📈 Evolução de Casos (Geral vs Por Categoria de Doença)")
                    
                    c_geral, c_comp = st.columns(2)
                    
                    with c_geral:
                        st.markdown("**Visão Geral (Total de Internações)**")
                        evolucao_geral = df_comparativo.groupby('ANO_CMPT').size().reset_index(name='TOTAL')
                        fig_geral = px.line(
                            evolucao_geral, x='ANO_CMPT', y='TOTAL', markers=True, text='TOTAL',
                            labels={'ANO_CMPT': 'Ano', 'TOTAL': 'Total'}
                        )
                        fig_geral.update_traces(textposition="top center", line_color='#ff7f0e', line_width=4, marker_size=10, textfont_size=14)
                        fig_geral.update_layout(template="plotly_dark", margin={"r":0,"t":10,"l":0,"b":0}, xaxis=dict(type='category'), height=400)
                        st.plotly_chart(fig_geral, use_container_width=True)
                        
                    with c_comp:
                        st.markdown("**Comparativo (Competição entre Doenças)**")
                        evolucao_comp = df_comparativo.groupby(['ANO_CMPT', 'CATEGORIA_DOENCA']).size().reset_index(name='TOTAL')
                        fig_comp = px.line(
                            evolucao_comp, x='ANO_CMPT', y='TOTAL', color='CATEGORIA_DOENCA', markers=True,
                            labels={'ANO_CMPT': 'Ano', 'TOTAL': 'Total', 'CATEGORIA_DOENCA': 'Doença'}
                        )
                        fig_comp.update_traces(line_width=3, marker_size=8)
                        # A legenda fica abaixo do gráfico para não esmagar as linhas
                        fig_comp.update_layout(
                            template="plotly_dark", margin={"r":0,"t":10,"l":0,"b":0}, 
                            xaxis=dict(type='category'), height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Sem dados suficientes para exibir a evolução.")
                
        else:
            st.info("O ficheiro 'coords_bairros_campinas.csv' não foi encontrado na pasta processed.")
