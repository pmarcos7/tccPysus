"""
API REST de Inferência para Previsão de Tempo de Permanência Hospitalar (RMC).
Coloque este arquivo na raiz do repositório `tccPysus`.

Como executar:
    python -m uvicorn api_permanencia:app --reload --port 8000

Interface gráfica Swagger:
    http://127.0.0.1:8000/docs
"""

from pathlib import Path
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(TRACKING_URI)

MODELO_REGISTRO_NOME = "Permanencia_RMC_Regressor"
MODEL_URI_CHAMPION = f"models:/{MODELO_REGISTRO_NOME}@champion"

app = FastAPI(
    title="API MLOps — Predição de Permanência Hospitalar SUS (RMC)",
    description=(
        "Serviço de inferência em tempo real integrado ao **MLflow Model Registry**. "
        "Estima os dias de internação (`DIAS_PERM`) para doenças respiratórias (Cap. J CID-10) "
        "nos 20 municípios da Região Metropolitana de Campinas."
    ),
    version="1.0.0",
)

# Habilita o CORS para permitir requisições diretas do painel HTML no navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AdmissaoHospitalarInput(BaseModel):
    IDADE: int = Field(
        ...,
        ge=0,
        le=120,
        description="Idade do paciente em anos completos.",
        examples=[68],
    )
    SEXO: str = Field(
        ...,
        description="Código de sexo no padrão SIH/DATASUS ('1' Masculino, '3' Feminino).",
        examples=["1"],
    )
    FAIXA_ETARIA: str = Field(
        ...,
        description="Faixa etária do projeto ('Crianca', 'Adolescente', 'Adulto', 'Idoso').",
        examples=["Idoso"],
    )
    ESTACAO: str = Field(
        ...,
        description="Estação climática ('Inverno' para meses 6 a 9, ou 'Outras').",
        examples=["Inverno"],
    )
    SUBCATEGORIA_CID: str = Field(
        ...,
        description="Subcategoria CID-10 ('Influenza/Pneumonia', 'DPOC/Asma', 'IRA Superior', 'IRA Aguda', 'Outras Respiratorias').",
        examples=["Influenza/Pneumonia"],
    )
    TIPO_ATENDIMENTO: str = Field(
        ...,
        description="Caráter do atendimento ('Urgencia' ou 'Eletiva').",
        examples=["Urgencia"],
    )
    NOME_MUNIC_ATENDIMENTO: str = Field(
        ...,
        description="Município onde ocorreu o atendimento dentro da RMC.",
        examples=["Campinas"],
    )


def carregar_modelo_producao():
    client = MlflowClient()
    try:
        versao_info = client.get_model_version_by_alias(MODELO_REGISTRO_NOME, "champion")
        modelo = mlflow.pyfunc.load_model(MODEL_URI_CHAMPION)
        return modelo, versao_info.version, versao_info.run_id
    except Exception:
        versoes = client.search_model_versions(f"name='{MODELO_REGISTRO_NOME}'")
        if not versoes:
            raise RuntimeError(
                f"Nenhum modelo '{MODELO_REGISTRO_NOME}' encontrado no MLflow. "
                "Execute primeiro: python permanencia_model_mlops.py"
            )
        ultima = sorted(versoes, key=lambda v: int(v.version), reverse=True)[0]
        uri_fallback = f"models:/{MODELO_REGISTRO_NOME}/{ultima.version}"
        modelo = mlflow.pyfunc.load_model(uri_fallback)
        return modelo, ultima.version, ultima.run_id


@app.get("/", summary="Rota raiz de boas-vindas e redirecionamento")
def raiz():
    return {
        "servico": "API MLOps Permanência Hospitalar SUS - RMC",
        "status": "online",
        "documentacao_swagger": "http://127.0.0.1:8000/docs",
    }


@app.get("/health", summary="Verifica integridade da API e versão do modelo no MLflow")
def health_check():
    try:
        _, versao, run_id = carregar_modelo_producao()
        return {
            "status": "online",
            "mlflow_tracking_uri": TRACKING_URI,
            "modelo_registrado": MODELO_REGISTRO_NOME,
            "versao_ativa": versao,
            "run_id_origem": run_id,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/permanencia", summary="Estima os dias de permanência hospitalar (DIAS_PERM)")
def prever_permanencia(dados: AdmissaoHospitalarInput):
    try:
        modelo, versao, run_id = carregar_modelo_producao()
        df_input = pd.DataFrame([dados.model_dump()])
        predicao = float(modelo.predict(df_input)[0])

        return {
            "previsao_dias_permanencia": round(predicao, 2),
            "previsao_arredondada_leito_dias": int(round(predicao)),
            "governanca_mlops": {
                "modelo_nome": MODELO_REGISTRO_NOME,
                "versao_consumida": versao,
                "alias": "@champion",
                "run_id": run_id,
            },
            "dados_paciente_recebidos": dados.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao executar inferência: {str(e)}")