import os
import gc
import glob
import time
import requests
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from utils import (
    MUNICIPIOS_RMC_NOMES,
    normalizar_nome,
    classificar_faixa_etaria,
    classificar_estacao,
    subcategoria_cid_respiratorio,
    limpar_cep,
)

COLUNAS_VITAIS = ['N_AIH', 'ANO_CMPT', 'MES_CMPT', 'IDADE', 'SEXO', 'MUNIC_RES', 'MUNIC_MOV', 'CEP', 'CNES', 'CAR_INT', 'DIAG_PRINC', 'DIAS_PERM', 'MORTE']
ANOS_ANALISE = {2022, 2023, 2024, 2025}
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
    if len(codigos_rmc) != len(MUNICIPIOS_RMC_NOMES):
        raise RuntimeError(
            "Não foi possível confirmar os códigos de todos os 20 municípios da RMC "
            "na API do IBGE. Verifique a conexão e tente novamente; o pipeline foi "
            "interrompido para não gerar uma base com recorte geográfico incompleto."
        )
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
        codigos_residencia = (
            df_lote['MUNIC_RES']
            .astype('string')
            .str.replace(r'\.0$', '', regex=True)
            .str[:6]
        )
        df_lote = df_lote[codigos_residencia.isin(codigos_rmc)]
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
        codigo_atendimento = (
            df_gold['MUNIC_MOV']
            .astype('string')
            .str.replace(r'\.0$', '', regex=True)
            .str[:6]
        )
    else:
        # Não inferir o município de atendimento pelo número da AIH: o campo
        # MUNIC_MOV ausente deve permanecer desconhecido.
        codigo_atendimento = pd.Series(pd.NA, index=df_gold.index, dtype='string')

    codigo_residencia = (
        df_gold['MUNIC_RES']
        .astype('string')
        .str.replace(r'\.0$', '', regex=True)
        .str[:6]
    )
    df_gold['MUNIC_ATENDIMENTO'] = codigo_atendimento
    df_gold['NOME_MUNIC_RES'] = codigo_residencia.map(dic_nomes).fillna('Nao Informado')

    nomes_atendimento = codigo_atendimento.map(dic_nomes)
    atendimento_conhecido = nomes_atendimento.notna()
    atendimento_na_rmc = codigo_atendimento.isin(codigos_rmc)
    df_gold['ATENDIDO_FORA_RMC'] = pd.Series(pd.NA, index=df_gold.index, dtype='Int64')
    df_gold.loc[atendimento_conhecido, 'ATENDIDO_FORA_RMC'] = (
        ~atendimento_na_rmc.loc[atendimento_conhecido]
    ).astype('int64')

    residencia_e_atendimento_conhecidos = (
        codigo_residencia.isin(dic_nomes) & atendimento_conhecido
    )
    df_gold['ATENDIDO_FORA'] = pd.Series(pd.NA, index=df_gold.index, dtype='Int64')
    df_gold.loc[residencia_e_atendimento_conhecidos, 'ATENDIDO_FORA'] = (
        codigo_residencia.loc[residencia_e_atendimento_conhecidos]
        != codigo_atendimento.loc[residencia_e_atendimento_conhecidos]
    ).astype('int64')

    df_gold['NOME_MUNIC_ATENDIMENTO'] = nomes_atendimento.fillna('Nao Informado')
    atendido_fora_rmc = df_gold['ATENDIDO_FORA_RMC'].eq(1)
    df_gold.loc[atendido_fora_rmc, 'NOME_MUNIC_ATENDIMENTO'] = 'Fora da RMC'
    # O destino externo é mantido somente como indicador agregado, sem conservar
    # o código ou o nome da cidade fora da RMC nos arquivos processados.
    df_gold.loc[atendido_fora_rmc, 'MUNIC_ATENDIMENTO'] = pd.NA

    # ViaCEP
    df_gold['CEP_LIMPO'] = limpar_cep(df_gold['CEP'])
    residencia_campinas = df_gold['MUNIC_RES'].astype(str).str.startswith('350950')
    mask_campinas = residencia_campinas & df_gold['ATENDIDO_FORA_RMC'].eq(0)
    ceps_campinas = df_gold.loc[mask_campinas, 'CEP_LIMPO'].dropna().unique().tolist()
    
    arq_cep_cache = os.path.join(caminho_processed, 'cep_bairro_cache.parquet')
    df_cache = consultar_viacep_com_cache(ceps_campinas, arq_cep_cache)
    mapa_cep_bairro = df_cache.set_index('CEP')['NOME_BAIRRO']

    df_gold['NOME_BAIRRO'] = 'Fora de Campinas'
    df_gold.loc[
        residencia_campinas & df_gold['ATENDIDO_FORA_RMC'].eq(1), 'NOME_BAIRRO'
    ] = 'Fora do recorte de atendimento na RMC'
    df_gold.loc[mask_campinas, 'NOME_BAIRRO'] = df_gold.loc[mask_campinas, 'CEP_LIMPO'].map(mapa_cep_bairro).fillna('Bairro Nao Informado/Invalido')
    df_gold = df_gold.drop(columns=['CEP', 'CEP_LIMPO'], errors='ignore')

    df_gold['MORTE'] = pd.to_numeric(df_gold['MORTE'], errors='coerce').fillna(0)
    # Registros de atendimento fora da região são reduzidos a contagens por ano
    # e município de residência. Não carregamos diagnósticos ou destinos externos
    # para as análises de permanência e fluxo interno.
    fluxo_saidas_rmc = (
        df_gold.loc[df_gold['ATENDIDO_FORA_RMC'].eq(1)]
        .groupby(['ANO_CMPT', 'MUNIC_RES', 'NOME_MUNIC_RES'], dropna=False)
        .agg(
            INTERNACOES_FORA_RMC=('ATENDIDO_FORA_RMC', 'size'),
        )
        .reset_index()
    )
    df_eda = df_gold[df_gold['ATENDIDO_FORA_RMC'].eq(0)].drop(
        columns=['ATENDIDO_FORA_RMC'], errors='ignore'
    ).copy()
    df_ml = df_eda[df_eda['MORTE'] != 1].drop(columns=['MORTE'], errors='ignore').copy()

    df_eda.to_parquet(os.path.join(caminho_processed, 'gold_resp_eda.parquet'), index=False)
    df_ml.to_parquet(os.path.join(caminho_processed, 'gold_resp_ml.parquet'), index=False)
    fluxo_saidas_rmc.to_parquet(
        os.path.join(caminho_processed, 'gold_resp_fluxo.parquet'), index=False
    )
    total_fora_rmc = int(fluxo_saidas_rmc['INTERNACOES_FORA_RMC'].sum())
    total_atendidos_rmc = len(df_eda)
    print(
        f"Recorte RMC: {total_atendidos_rmc} internações atendidas na região; "
        f"{total_fora_rmc} moradores atendidos fora (destinos agregados)."
    )
