import os
import gc
import glob
import time
import requests
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from utils import normalizar_nome, classificar_faixa_etaria, classificar_estacao, subcategoria_cid_respiratorio, limpar_cep

COLUNAS_VITAIS = ['N_AIH', 'ANO_CMPT', 'MES_CMPT', 'IDADE', 'SEXO', 'MUNIC_RES', 'MUNIC_MOV', 'CEP', 'CNES', 'CAR_INT', 'DIAG_PRINC', 'DIAS_PERM', 'MORTE']
ANOS_ANALISE = {2022, 2023, 2024, 2025}
MUNICIPIOS_RMC_NOMES = ['Americana', 'Artur Nogueira', 'Campinas', 'Cosmópolis', 'Engenheiro Coelho', 'Holambra', 'Hortolândia', 'Indaiatuba', 'Itatiba', 'Jaguariúna', 'Monte Mor', 'Morungaba', 'Nova Odessa', 'Paulínia', 'Pedreira', "Santa Bárbara D'Oeste", 'Santo Antônio de Posse', 'Sumaré', 'Valinhos', 'Vinhedo']

def obter_codigos_rmc():
    url_ibge = 'https://servicodados.ibge.gov.br/api/v1/localidades/municipios'
    try:
        resp = requests.get(url_ibge, timeout=30)
        resp.raise_for_status()
        municipios = resp.json()
        dic_nomes = {str(m['id'])[:6]: m['nome'] for m in municipios}
    except requests.RequestException:
        dic_nomes = {}
        
    nomes_rmc_norm = {normalizar_nome(n) for n in MUNICIPIOS_RMC_NOMES}
    codigos_rmc = {cod for cod, nome in dic_nomes.items() if cod.startswith('35') and normalizar_nome(nome) in nomes_rmc_norm}
    return codigos_rmc, dic_nomes

def consultar_viacep_com_cache(ceps_unicos, arq_cep_cache):
    if os.path.exists(arq_cep_cache):
        df_cache = pd.read_parquet(arq_cep_cache)
    else:
        df_cache = pd.DataFrame(columns=['CEP', 'NOME_BAIRRO'])

    ja_conhecidos = set(df_cache['CEP'])
    pendentes = [c for c in ceps_unicos if c not in ja_conhecidos]

    novos = []
    for cep in tqdm(pendentes, desc='ViaCEP', unit='cep'):
        try:
            resp = requests.get(f'https://viacep.com.br/ws/{cep}/json/', timeout=5)
            if resp.status_code == 200:
                dados = resp.json()
                bairro = dados.get('bairro') if 'erro' not in dados else None
                novos.append({'CEP': cep, 'NOME_BAIRRO': bairro or 'Bairro Nao Informado'})
        except requests.RequestException:
            pass
        time.sleep(0.05)

    if novos:
        df_cache = pd.concat([df_cache, pd.DataFrame(novos)], ignore_index=True).drop_duplicates('CEP')
        df_cache.to_parquet(arq_cep_cache, index=False)

    return df_cache

def executar_pipeline(caminho_raw, caminho_processed):
    os.makedirs(caminho_processed, exist_ok=True)
    codigos_rmc, dic_nomes = obter_codigos_rmc()
    
    arquivos = sorted(glob.glob(os.path.join(caminho_raw, '*.parquet')))
    arquivos = [a for a in arquivos if any(str(ano) in os.path.basename(a) for ano in ANOS_ANALISE)]
    
    lotes = []
    for arquivo in tqdm(arquivos, desc='Lotes Silver', unit='mes'):
        schema_cols = pq.read_schema(arquivo).names
        cols_leitura = [c for c in COLUNAS_VITAIS if c in schema_cols]
        df_lote = pd.read_parquet(arquivo, columns=cols_leitura)

        df_lote['ANO_CMPT'] = pd.to_numeric(df_lote['ANO_CMPT'], errors='coerce')
        df_lote = df_lote[df_lote['ANO_CMPT'].isin(ANOS_ANALISE)]
        df_lote = df_lote[df_lote['MUNIC_RES'].astype(str).str[:6].isin(codigos_rmc)]
        df_lote['DIAG_PRINC'] = df_lote['DIAG_PRINC'].astype(str).str.strip()
        df_lote = df_lote[df_lote['DIAG_PRINC'].str[0].str.upper() == 'J']
        df_lote = df_lote.drop_duplicates(subset=['N_AIH'])
        df_lote['DIAS_PERM'] = pd.to_numeric(df_lote['DIAS_PERM'], errors='coerce')
        df_lote = df_lote[df_lote['DIAS_PERM'].between(1, 60)].copy()

        if len(df_lote) > 0:
            lotes.append(df_lote)
        del df_lote
        gc.collect()

    df_gold = pd.concat(lotes, ignore_index=True)
    del lotes
    gc.collect()

    # Engenharia de Features (Gold)
    df_gold['IDADE'] = pd.to_numeric(df_gold['IDADE'], errors='coerce').fillna(0).astype(int)
    df_gold['FAIXA_ETARIA'] = df_gold['IDADE'].apply(classificar_faixa_etaria)
    df_gold['ESTACAO'] = df_gold['MES_CMPT'].apply(classificar_estacao)
    df_gold['SUBCATEGORIA_CID'] = df_gold['DIAG_PRINC'].astype(str).apply(subcategoria_cid_respiratorio)
    df_gold['TIPO_ATENDIMENTO'] = df_gold['CAR_INT'].astype(str).str.zfill(2).map({'01': 'Eletiva', '02': 'Urgencia'}).fillna('Outros')

    if 'MUNIC_MOV' in df_gold.columns:
        df_gold['MUNIC_ATENDIMENTO'] = df_gold['MUNIC_MOV'].astype(str).str[:6]
    else:
        df_gold['MUNIC_ATENDIMENTO'] = df_gold['N_AIH'].astype(str).str[:6]

    df_gold['NOME_MUNIC_RES'] = df_gold['MUNIC_RES'].astype(str).str[:6].map(dic_nomes).fillna('Nao Informado')
    df_gold['NOME_MUNIC_ATENDIMENTO'] = df_gold['MUNIC_ATENDIMENTO'].map(dic_nomes).fillna('Nao Informado')
    df_gold['ATENDIDO_FORA'] = (df_gold['MUNIC_RES'].astype(str).str[:6] != df_gold['MUNIC_ATENDIMENTO']).astype(int)

    # ViaCEP
    df_gold['CEP_LIMPO'] = limpar_cep(df_gold['CEP'])
    mask_campinas = df_gold['MUNIC_RES'].astype(str).str.startswith('350950')
    ceps_campinas = df_gold.loc[mask_campinas, 'CEP_LIMPO'].dropna().unique().tolist()
    
    arq_cep_cache = os.path.join(caminho_processed, 'cep_bairro_cache.parquet')
    df_cache = consultar_viacep_com_cache(ceps_campinas, arq_cep_cache)
    mapa_cep_bairro = df_cache.set_index('CEP')['NOME_BAIRRO']

    df_gold['NOME_BAIRRO'] = 'Fora de Campinas'
    df_gold.loc[mask_campinas, 'NOME_BAIRRO'] = df_gold.loc[mask_campinas, 'CEP_LIMPO'].map(mapa_cep_bairro).fillna('Bairro Nao Informado/Invalido')
    df_gold = df_gold.drop(columns=['CEP', 'CEP_LIMPO', 'CNES'], errors='ignore')

    # Separação EDA e ML
    df_eda = df_gold.copy()
    df_gold['MORTE'] = pd.to_numeric(df_gold['MORTE'], errors='coerce').fillna(0)
    df_ml = df_gold[df_gold['MORTE'] != 1].drop(columns=['MORTE']).copy()

    df_eda.to_parquet(os.path.join(caminho_processed, 'gold_resp_eda.parquet'), index=False)
    df_ml.to_parquet(os.path.join(caminho_processed, 'gold_resp_ml.parquet'), index=False)