# Análise de internações respiratórias na Região Metropolitana de Campinas

Projeto acadêmico que analisa internações por doenças do aparelho respiratório (capítulo J da CID-10) com dados do SIH/SUS. O trabalho reúne análise exploratória, modelos de previsão do tempo de permanência e análise de deslocamentos para atendimento.

## Escopo geográfico e período

- O recorte principal é a Região Metropolitana de Campinas (RMC), formada por 20 municípios.
- Municípios considerados: Americana, Artur Nogueira, Campinas, Cosmópolis, Engenheiro Coelho, Holambra, Hortolândia, Indaiatuba, Itatiba, Jaguariúna, Monte Mor, Morungaba, Nova Odessa, Paulínia, Pedreira, Santa Bárbara d'Oeste, Santo Antônio de Posse, Sumaré, Valinhos e Vinhedo.
- As análises de permanência, comparações entre hospitais e deslocamentos detalhados consideram internações atendidas dentro da RMC.
- Internações de moradores da RMC atendidos fora da região entram somente em uma contagem agregada por ano e município de residência. O projeto não divulga o nome do destino externo.
- O período analisado é 2022–2025. Para avaliar os modelos de permanência, o treino usa 2022–2024 e o teste usa 2025.
- Cada linha representa uma internação (AIH); as contagens do projeto são de internações, não necessariamente de pessoas únicas.

## O que o projeto produz

1. **Pipeline de dados:** filtra internações respiratórias de residentes da RMC, remove duplicidades/outliers definidos pelo projeto e prepara variáveis para as análises.
2. **Permanência hospitalar:** compara modelos de regressão para estimar dias de internação e gera previsões e métricas de desempenho para 2025.
3. **Perfil e comparação por grupo:** resume a permanência por doença, faixa etária e tipo de atendimento; compara desempenho dos modelos nesses grupos.
4. **Comparação por hospital:** agrupa os resultados por CNES, doença e faixa etária, mostrando média observada e prevista. Só apresenta grupos com pelo menos 30 internações.
5. **Deslocamentos e evasão:** descreve atendimentos entre municípios da RMC. As saídas da região são contabilizadas em conjunto, sem indicar município externo.
6. **Dashboards:** `dashboard_permanencia.py` apresenta modelos, hospitais, perfil, previsões por grupo e deslocamentos; `dashboard.py` apresenta mapas e informações gerais do fluxo e das internações.

As diferenças entre médias observadas e previstas são associações nos dados e não demonstram que um hospital ou município cause internações mais longas ou curtas.

## Estrutura do repositório

```text
tccPysus/
├── TCC_Datasus/data/
│   ├── raw/                 # Parquets de entrada do SIH/DATASUS
│   └── processed/           # Bases geradas pelo pipeline
├── resultados_permanencia/  # CSVs usados pelos modelos e dashboard
├── data_pipeline.py         # Leitura, filtros e preparação dos dados
├── main.py                  # Execução das etapas do projeto
├── permanencia_model.py     # Modelo de permanência e comparação por hospital
├── analise_permanencia_rmc.py # Métricas, perfil e previsões por grupo
├── evasao_model.py          # Modelo de deslocamento/evasão
├── eda_graficos.py          # Gráficos exploratórios
├── dashboard_permanencia.py # Dashboard de permanência
├── dashboard.py             # Dashboard de fluxo e mapas
├── utils.py                 # Funções e lista dos municípios da RMC
└── requirements.txt         # Dependências Python
```

Os arquivos de dados são grandes e ficam fora do Git. Por isso, depois de clonar o código, cada integrante precisa receber os Parquets do grupo ou obtê-los na fonte indicada pelo trabalho e colocá-los nas pastas corretas. Não envie bases grandes em commits do repositório.

## Como preparar o ambiente no Windows

Abra a pasta `tccPysus` no VS Code e abra um terminal PowerShell dentro dela. Confira se o terminal está na pasta que contém `main.py` e `requirements.txt`.

1. Instale o Python (recomendado: Python 3.12) e crie um ambiente virtual:

```powershell
py -3.12 -m venv .venv
```

2. Instale as dependências dentro do ambiente virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Não é necessário ativar o ambiente virtual: os comandos abaixo chamam diretamente o Python instalado nele.

## Onde colocar os dados

- Parquets brutos mensais do SIH/SUS: `TCC_Datasus/data/raw/`.
- O pipeline procura arquivos `.parquet` cujo nome contenha o ano entre 2022 e 2025.
- As bases processadas são gravadas em `TCC_Datasus/data/processed/`.
- Os CSVs de resultados são gravados em `resultados_permanencia/`.

O pipeline consulta a API do IBGE para confirmar os 20 municípios da RMC e usa o ViaCEP para complementar bairros de Campinas; portanto, a etapa de processamento precisa de conexão com a internet. Se o grupo já compartilhou as bases processadas atualizadas, não precisa rodar o pipeline novamente para abrir os dashboards.

## Como executar

Execute os comandos a seguir no terminal do VS Code, na pasta principal `tccPysus`.

### 1. Processar os dados

Faça isso quando tiver os Parquets brutos em `TCC_Datasus/data/raw/` ou quando os dados brutos forem atualizados:

```powershell
.\.venv\Scripts\python.exe .\main.py --step pipeline
```

O pipeline cria/atualiza as bases em `TCC_Datasus/data/processed/`. A execução pode demorar, especialmente se houver muitos arquivos ou CEPs ainda não presentes no cache.

### 2. Gerar análises de permanência e comparação por hospital

```powershell
.\.venv\Scripts\python.exe .\main.py --step permanencia
.\.venv\Scripts\python.exe .\analise_permanencia_rmc.py
```

O primeiro comando treina o modelo de permanência e gera `comparacao_hospitais_2025.csv`. O segundo compara os modelos e atualiza métricas, previsões, perfil por grupo e comparação por município em `resultados_permanencia/`.

### 3. Executar as outras análises (opcional)

```powershell
.\.venv\Scripts\python.exe .\main.py --step evasao
.\.venv\Scripts\python.exe .\main.py --step graficos
```

Para rodar pipeline, permanência, evasão e gráficos em sequência:

```powershell
.\.venv\Scripts\python.exe .\main.py --step all
```

`--step all` não substitui `analise_permanencia_rmc.py`; execute esse script à parte para atualizar as análises de perfil e comparação dos modelos mostradas no dashboard de permanência.

### 4. Abrir os dashboards

Abra um dashboard por vez. O terminal ficará ocupado enquanto o aplicativo estiver rodando; use `Ctrl+C` no terminal para encerrá-lo.

Dashboard de permanência, modelos, grupos, hospitais e deslocamentos:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\dashboard_permanencia.py
```

Dashboard de mapas e fluxo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\dashboard.py
```

O Streamlit mostrará um endereço local, normalmente `http://localhost:8501`. Abra esse endereço no navegador. Se a porta estiver ocupada por outro dashboard, o Streamlit escolherá outra e informará o endereço no terminal.

## Arquivos gerados mais importantes

Em `TCC_Datasus/data/processed/`:

- `gold_resp_eda.parquet`: internações respiratórias para as análises exploratórias e o dashboard, atendidas dentro da RMC.
- `gold_resp_ml.parquet`: base de modelagem de permanência, atendida dentro da RMC e sem óbitos.
- `gold_resp_fluxo.parquet`: contagens agregadas de internações de residentes da RMC atendidos fora dela; não contém os destinos externos.

Em `resultados_permanencia/`:

- `metricas_modelos.csv`: métricas dos modelos no teste de 2025.
- `previsoes_2025.csv`: previsões por internação do conjunto de teste.
- `metricas_por_grupo_2025.csv`: perfil e desempenho por doença, faixa etária e tipo de atendimento.
- `comparacao_por_municipio_2025.csv`: resumo por município de atendimento dentro da RMC.
- `comparacao_hospitais_2025.csv`: resumo por estabelecimento (CNES), doença e faixa etária dentro da RMC.

## Atualizar o código pelo GitHub Desktop

1. No GitHub Desktop, escolha a branch de trabalho combinada com o grupo.
2. Clique em **Fetch origin** e depois em **Pull origin** para baixar as alterações mais recentes dessa branch.
3. No VS Code, salve os arquivos e rode novamente os comandos das análises que foram alteradas.
4. Confira as mudanças na aba **Changes** do GitHub Desktop. Faça commit e push somente quando o grupo concordar com a atualização.

Se houver alterações locais ainda não salvas em commit, o GitHub Desktop pode impedir a troca de branch ou a atualização. Nesse caso, revise e salve essas mudanças antes de continuar.
