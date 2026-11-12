# Guia de Execução — MLOps SUS / DATASUS (RMC)

Este guia contém as instruções para executar o pipeline de **MLOps** (rastreamento de experimentos, governança no Model Registry e API de inferência) acoplado aos dados do TCC sobre internações respiratórias na Região Metropolitana de Campinas.

---

## 1. Onde colocar os arquivos no repositório

Coloque os arquivos criados diretamente na pasta raiz do seu projeto `tccPysus`:

```text
tccPysus/
├── permanencia_model_mlops.py   <-- Script de treino com MLflow
├── api_permanencia.py           <-- API FastAPI de predição
├── README_MLOPS.md              <-- Este guia de instruções
├── main.py
├── utils.py
├── analise_permanencia_rmc.py
├── TCC_Datasus/
│   └── data/processed/gold_resp_ml.parquet
└── .venv/
```

---

## 2. Instalação das dependências

Abra o terminal PowerShell do VS Code dentro da pasta `tccPysus` e instale as bibliotecas necessárias no ambiente virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install mlflow fastapi uvicorn
```

---

## 3. Passo 1 — Executar o Treino com Rastreamento e Registro de Modelo

Execute o script de treino:

```powershell
.\.venv\Scripts\python.exe .\permanencia_model_mlops.py
```

> **Dica de contingência:** Caso esteja testando em um computador que não tenha o arquivo `gold_resp_ml.parquet` baixado, adicione a flag de fallback para rodar a esteira com dados simulados:
> ```powershell
> .\.venv\Scripts\python.exe .\permanencia_model_mlops.py --demo-fallback
> ```

O que acontece durante a execução:
1. O script treina o **Baseline (média)**, o **HistGradientBoostingRegressor** e o **RandomForestRegressor** (com e sem município).
2. As métricas ($MAE$, $RMSE$ e $R^2$) de cada modelo são salvas no banco relacional `mlflow.db`.
3. O modelo de menor erro ($MAE$) é automaticamente promovido para o catálogo central `Permanencia_RMC_Regressor` com o rótulo `@champion`.

---

## 4. Passo 2 — Abrir o Painel Visual do MLflow (Para o Seminário)

Abra uma **segunda aba de terminal** no VS Code e inicie a interface gráfica:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Acesse no navegador:
👉 **http://127.0.0.1:5000**

**O que mostrar na apresentação:**
* Vá em **Experiments** > clique no experimento `SUS_RMC_Permanencia_Hospitalar`.
* Clique na aba lateral **`Training runs`** (ou no botão superior **`Model training`**).
* Veja a tabela com as 4 execuções e clique no botão de gráficos para comparar o $MAE$, $RMSE$ e $R^2$ lado a lado.
* No menu lateral, acesse **`Models`** / **`AI Registry`** e mostre o modelo `Permanencia_RMC_Regressor` com a versão `1` e o alias `@champion`.

---

## 5. Passo 3 — Subir a API de Predição Online

Abra uma **terceira aba de terminal** no VS Code e inicie o serviço web:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_permanencia:app --reload --port 8000
```

Acesse a interface interativa no navegador:
👉 **http://127.0.0.1:8000/docs**

**Como testar a predição ao vivo:**
1. Clique no endpoint **`POST /predict/permanencia`**.
2. Clique no botão **Try it out**.
3. O corpo da requisição já vem pré-preenchido com um exemplo:
   ```json
   {
     "IDADE": 68,
     "SEXO": "1",
     "FAIXA_ETARIA": "Idoso",
     "ESTACAO": "Inverno",
     "SUBCATEGORIA_CID": "Influenza/Pneumonia",
     "TIPO_ATENDIMENTO": "Urgencia",
     "NOME_MUNIC_ATENDIMENTO": "Campinas"
   }
   ```
4. Clique em **Execute**.
5. A API consultará dinamicamente o modelo `@champion` no MLflow e devolverá a resposta imediata:
   ```json
   {
     "previsao_dias_permanencia": 8.46,
     "previsao_arredondada_leito_dias": 8,
     "governanca_mlops": {
       "modelo_nome": "Permanencia_RMC_Regressor",
       "versao_consumida": "1",
       "alias": "@champion",
       "run_id": "6c346d63d2204dcabbc74bbbbb08ff91"
     }
   }
   ```