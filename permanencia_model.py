import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import MUNICIPIOS_RMC_NORMALIZADOS, normalizar_nome

def treinar_modelo_permanencia(caminho_gold_ml):
    df_ml = pd.read_parquet(caminho_gold_ml)

    # Comparação de permanência restrita a atendimentos feitos dentro da RMC.
    df_ml = df_ml[
        df_ml['NOME_MUNIC_ATENDIMENTO']
        .map(normalizar_nome)
        .isin(MUNICIPIOS_RMC_NORMALIZADOS)
    ].copy()
    
    # ATENDIDO_FORA removido para evitar leakage
    features_cat = ['FAIXA_ETARIA', 'ESTACAO', 'SUBCATEGORIA_CID', 'TIPO_ATENDIMENTO']
    features_num = ['IDADE']
    
    X = df_ml[features_num + features_cat]
    y = df_ml['DIAS_PERM']
    
    mask_train = df_ml['ANO_CMPT'] <= 2024
    mask_test = df_ml['ANO_CMPT'] == 2025
    
    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), features_cat)],
        remainder='passthrough'
    )
    
    # Baseline
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    preds_dummy = dummy.predict(X_test)
    print(f"[Pilar 1] Baseline MAE: {mean_absolute_error(y_test, preds_dummy):.3f}")

    # Modelo HGBR
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=42))
    ])
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
            # Comparação por estabelecimento, doença e faixa etária
    df_teste_hospital = df_ml.loc[
        mask_test,
        ['CNES', 'NOME_MUNIC_ATENDIMENTO', 'SUBCATEGORIA_CID', 'FAIXA_ETARIA', 'DIAS_PERM']
    ].copy()

    df_teste_hospital['DIAS_PREVISTOS'] = preds

    comparacao_hospitais = (
        df_teste_hospital
        .groupby(['CNES', 'NOME_MUNIC_ATENDIMENTO', 'SUBCATEGORIA_CID', 'FAIXA_ETARIA'], dropna=False)
        .agg(
            N_INTERNACOES=('DIAS_PERM', 'size'),
            MEDIA_OBSERVADA_DIAS=('DIAS_PERM', 'mean'),
            MEDIA_PREVISTA_DIAS=('DIAS_PREVISTOS', 'mean')
        )
        .reset_index()
    )

    comparacao_hospitais['DIF_OBSERVADO_PREVISTO_DIAS'] = (
        comparacao_hospitais['MEDIA_OBSERVADA_DIAS']
        - comparacao_hospitais['MEDIA_PREVISTA_DIAS']
    )

    comparacao_hospitais = comparacao_hospitais[
        comparacao_hospitais['N_INTERNACOES'] >= 30
    ].sort_values(
        ['SUBCATEGORIA_CID', 'FAIXA_ETARIA', 'DIF_OBSERVADO_PREVISTO_DIAS']
    )

    from pathlib import Path
    pasta_resultados = Path('resultados_permanencia')
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    arquivo_comparacao = pasta_resultados / 'comparacao_hospitais_2025.csv'
    comparacao_hospitais.to_csv(
        arquivo_comparacao,
        index=False,
        encoding='utf-8-sig'
    )

    print(f'Comparação por hospital salva em: {arquivo_comparacao}')
    print(f'Grupos com pelo menos 30 internações: {len(comparacao_hospitais)}')
    
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    
    print(f"[Pilar 1] HistGradientBoostingRegressor -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
    return model
