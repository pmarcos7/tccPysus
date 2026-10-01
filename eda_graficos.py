import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA

def gerar_graficos_eda(caminho_gold_eda):
    print("A carregar dados para os gráficos EDA...")
    df_eda = pd.read_parquet(caminho_gold_eda)
    os.makedirs('plots', exist_ok=True)

    # ---------------------------------------------------------
    # 1. Distribuição de DIAS_PERM
    # ---------------------------------------------------------
    media = df_eda["DIAS_PERM"].mean()
    mediana = df_eda["DIAS_PERM"].median()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df_eda["DIAS_PERM"], bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(media, color="red", linestyle="--", label=f"Media={media:.1f} dias")
    ax.axvline(mediana, color="orange", linestyle="-", label=f"Mediana={mediana:.0f} dias")
    ax.set_title("Distribuicao de DIAS_PERM — Doencas Respiratorias RMC 2022-2025", fontsize=14, pad=12)
    ax.set_xlabel("Dias de Internacao", fontsize=12)
    ax.set_ylabel("Frequencia", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/01_distribuicao_dias_perm.png')
    plt.close()
    print("✔ Gráfico de Distribuição guardado em plots/01_distribuicao_dias_perm.png")

    # ---------------------------------------------------------
    # 2. Decomposição Sazonal (STL)
    # ---------------------------------------------------------
    _df_ts = df_eda.copy()
    _df_ts['DATA'] = pd.to_datetime(_df_ts['ANO_CMPT'].astype(str) + '-' + _df_ts['MES_CMPT'].astype(str).str.zfill(2) + '-01')
    serie = _df_ts.groupby('DATA').size().reset_index(name='INTERNACOES').sort_values('DATA').set_index('DATA')

    decomp = seasonal_decompose(serie['INTERNACOES'], model='additive', period=12, extrapolate_trend='freq')

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], color='steelblue', linewidth=2)
    axes[0].set_ylabel('Observado')
    axes[0].set_title('Decomposicao Sazonal -- Internacoes Respiratorias (RMC 2022-2025)', fontsize=14)
    
    decomp.trend.plot(ax=axes[1], color='darkorange', linewidth=2)
    axes[1].set_ylabel('Tendencia')
    
    decomp.seasonal.plot(ax=axes[2], color='teal', linewidth=2)
    axes[2].set_ylabel('Sazonalidade')
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    decomp.resid.plot(ax=axes[3], color='gray', linewidth=1, alpha=0.7)
    axes[3].set_ylabel('Residuo')
    axes[3].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('plots/02_decomposicao_sazonal.png')
    plt.close()
    print("✔ Gráfico de Sazonalidade guardado em plots/02_decomposicao_sazonal.png")

    # ---------------------------------------------------------
    # 3. PCA 2D - Estrutura Latente
    # ---------------------------------------------------------
    cols_num = ['IDADE', 'DIAS_PERM', 'MES_CMPT', 'ATENDIDO_FORA']
    cols_cat = ['FAIXA_ETARIA', 'ESTACAO', 'TIPO_ATENDIMENTO', 'SUBCATEGORIA_CID']
    
    # Valida as colunas existentes no parquet
    cols_num = [c for c in cols_num if c in df_eda.columns]
    cols_cat = [c for c in cols_cat if c in df_eda.columns]
    
    amostra_pca = df_eda[cols_num + cols_cat].dropna().sample(n=min(20000, len(df_eda)), random_state=42)
    
    enc_ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc_ord.fit_transform(amostra_pca[cols_cat])
    X_all = np.hstack([amostra_pca[cols_num].values, X_cat])
    X_scaled = StandardScaler().fit_transform(X_all)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, c=amostra_pca['DIAS_PERM'], cmap='viridis')
    plt.colorbar(scatter, label='Dias de Permanência')
    ax.set_title(f'PCA 2D (PC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%)', fontsize=14)
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    plt.tight_layout()
    plt.savefig('plots/03_pca_2d.png')
    plt.close()
    print("✔ Gráfico de PCA guardado em plots/03_pca_2d.png")

if __name__ == "__main__":
    gerar_graficos_eda('./TCC_Datasus/data/processed/gold_resp_eda.parquet')