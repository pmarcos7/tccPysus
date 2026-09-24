import unicodedata
import pandas as pd
import numpy as np

def normalizar_nome(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return s.upper().strip()

def classificar_faixa_etaria(idade):
    try:
        i = int(idade)
        if i <= 11: return 'Crianca'
        if i <= 17: return 'Adolescente'
        if i <= 59: return 'Adulto'
        return 'Idoso'
    except: return 'Desconhecido'

def classificar_estacao(mes):
    try: return 'Inverno' if int(mes) in (6, 7, 8, 9) else 'Outras'
    except: return 'Desconhecida'

def subcategoria_cid_respiratorio(cid):
    if not isinstance(cid, str) or len(cid) < 3:
        return 'Outras Respiratorias'
    try:
        num = int(cid[1:3])
    except ValueError:
        return 'Outras Respiratorias'
    if num <= 6:  return 'IRA Superior'
    if num <= 22: return 'Influenza/Pneumonia'
    if num <= 39: return 'IRA Aguda'
    if num <= 47: return 'DPOC/Asma'
    return 'Outras Respiratorias'

def limpar_cep(serie_cep):
    s = serie_cep.copy()
    if pd.api.types.is_float_dtype(s):
        s = s.astype('Int64')
    s = s.astype(str).str.replace(r'\.0$', '', regex=True)
    s = s.str.replace(r'[^0-9]', '', regex=True).str.zfill(8)
    invalidos = {'00000000', '99999999', '11111111'}
    s = s.where(~s.isin(invalidos) & (s.str.len() == 8), np.nan)
    return s