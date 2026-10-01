import argparse
from data_pipeline import executar_pipeline
from permanencia_model import treinar_modelo_permanencia
from evasao_model import treinar_modelo_evasao
from eda_graficos import gerar_graficos_eda

CAMINHO_RAW = './TCC_Datasus/data/raw'
CAMINHO_PROCESSED = './TCC_Datasus/data/processed'
ARQ_GOLD_ML = f"{CAMINHO_PROCESSED}/gold_resp_ml.parquet"

def main():
    parser = argparse.ArgumentParser(description="Pipeline TCC DATASUS")
    parser.add_argument('--step', type=str, choices=['pipeline', 'permanencia', 'evasao', 'graficos', 'all'], default='all')
    args = parser.parse_args()

    if args.step in ['pipeline', 'all']:
        print("--- Iniciando Processamento de Dados ---")
        executar_pipeline(CAMINHO_RAW, CAMINHO_PROCESSED)
        
    if args.step in ['permanencia', 'all']:
        print("--- Iniciando Pilar 1: Permanência Hospitalar ---")
        modelo_regressao = treinar_modelo_permanencia(ARQ_GOLD_ML)
        
    if args.step in ['evasao', 'all']:
        print("--- Iniciando Pilar 2: Evasão Hospitalar ---")
        modelo_classificacao = treinar_modelo_evasao(ARQ_GOLD_ML)
    
    if args.step in ['graficos', 'all']:
        print("--- Iniciando Geração de Gráficos (EDA) ---")
        gerar_graficos_eda(f"{CAMINHO_PROCESSED}/gold_resp_eda.parquet")

if __name__ == "__main__":
    main()