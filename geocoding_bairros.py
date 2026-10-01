import pandas as pd
import time
from geopy.geocoders import Nominatim

# Inicializar o geocodificador (o user_agent é obrigatório pela API gratuita)
geolocator = Nominatim(user_agent="tcc_sus_campinas")

def obter_coordenadas_bairro(nome_bairro):
    """Procura a latitude e longitude de um bairro específico em Campinas."""
    # A string de busca precisa ser bem específica para a API não confundir com outras cidades
    query = f"{nome_bairro}, Campinas, São Paulo, Brasil"
    try:
        location = geolocator.geocode(query, timeout=10)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None, None

# Exemplo de como gerar o seu dicionário de Bairros:
# 1. Carregue os seus dados do TCC
df = pd.read_parquet('./TCC_Datasus/data/processed/gold_resp_eda.parquet')

# 2. Isole apenas os bairros de Campinas válidos
bairros_campinas = df[df['NOME_BAIRRO'] != 'Fora de Campinas']['NOME_BAIRRO'].dropna().unique()

coordenadas_bairros = []

print("A buscar coordenadas na API do OpenStreetMap...")
for bairro in bairros_campinas:
    if bairro == 'Bairro Nao Informado/Invalido':
        continue
        
    lat, lon = obter_coordenadas_bairro(bairro)
    coordenadas_bairros.append({'BAIRRO': bairro, 'LAT': lat, 'LON': lon})
    time.sleep(1.1) # Respeitar o limite da API pública (1 req/seg)

# 3. Guardar num ficheiro CSV para o Dashboard consumir instantaneamente
df_coords = pd.DataFrame(coordenadas_bairros)
df_coords.to_csv('./TCC_Datasus/data/processed/coords_bairros_campinas.csv', index=False)
print("Coordenadas guardadas com sucesso!")