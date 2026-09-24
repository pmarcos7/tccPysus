import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def treinar_modelo_evasao(caminho_gold_ml):
    df_ml = pd.read_parquet(caminho_gold_ml)
    
    features_cat = ['FAIXA_ETARIA', 'ESTACAO', 'SUBCATEGORIA_CID', 'TIPO_ATENDIMENTO']
    features_num = ['IDADE']
    
    X = df_ml[features_num + features_cat]
    y = df_ml['ATENDIDO_FORA']
    
    mask_train = df_ml['ANO_CMPT'] <= 2024
    mask_test = df_ml['ANO_CMPT'] == 2025
    
    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), features_cat)],
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    print(f"\n[Pilar 2] Random Forest AUC-ROC: {auc:.3f}")
    print("[Pilar 2] Classification Report (Teste 2025):")
    print(classification_report(y_test, preds))
    
    return model