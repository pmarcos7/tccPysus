import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def treinar_modelo_permanencia(caminho_gold_ml):
    df_ml = pd.read_parquet(caminho_gold_ml)
    
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
    
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    
    print(f"[Pilar 1] HistGradientBoostingRegressor -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")
    return model