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

1. **Pipeline de dados:** filtra internações respiratórias de residentes da RMC, remove duplicidades/outliers definidos pelo projeto e prepara variáveis para as análises, incluindo variáveis de gravidade clínica (`USOU_UTI` = passou pela UTI; `N_COMORBID` = número de comorbidades).
2. **Permanência hospitalar:** compara modelos de regressão para estimar dias de internação e gera previsões e métricas de desempenho para 2025. A comparação é feita contra um **baseline por subgrupo** (média de dias do perfil doença × faixa etária × tipo de atendimento), para não superestimar o valor do ML.
3. **Perfil e comparação por grupo:** resume a permanência por doença, faixa etária e tipo de atendimento; compara desempenho dos modelos nesses grupos e por segmento (urgência × eletiva).
4. **Comparação por hospital:** agrupa os resultados por CNES, doença e faixa etária, mostrando média observada e prevista. Só apresenta grupos com pelo menos 30 internações.
5. **Deslocamentos e evasão:** descreve atendimentos entre municípios da RMC. As saídas da região são contabilizadas em conjunto, sem indicar município externo.
6. **Simulador de permanência (2026):** `previsao.py` treina um modelo de produção com todo o histórico (2022–2025) usando **só variáveis conhecidas na admissão** (idade, faixa etária, estação, subcategoria de CID e tipo de atendimento) e estima os dias de internação de novos pacientes. Fica disponível também na aba "Simulador 2026" do dashboard de permanência.
7. **Dashboards:** `dashboard_permanencia.py` apresenta modelos, hospitais, perfil, previsões por grupo, simulador 2026 e deslocamentos; `dashboard.py` apresenta mapas e informações gerais do fluxo e das internações.

As diferenças entre médias observadas e previstas são associações nos dados e não demonstram que um hospital ou município cause internações mais longas ou curtas.

### Observação sobre os modelos (achado do projeto)

Com variáveis apenas demográficas/administrativas o ML empata com o baseline por subgrupo (R² ~0,08). Ao adicionar a gravidade clínica **`USOU_UTI`**, o erro em urgências cai ~9% e o R² triplica (0,06 → 0,21): só então o modelo supera o baseline. Como `USOU_UTI` só é conhecido **durante** a internação, esse é um modelo **explicativo**, não de previsão na admissão — por isso o `previsao.py` (que prevê na admissão) **não** usa UTI. Os detalhes estão em `NOTAS_MODELAGEM.md`.

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
├── analise_permanencia_rmc.py # Métricas, perfil, segmentos e comparação dos modelos
├── previsao.py              # Modelo de produção e simulador de permanência (2026)
├── evasao_model.py          # Modelo de deslocamento/evasão
├── eda_graficos.py          # Gráficos exploratórios
├── dashboard_permanencia.py # Dashboard de permanência (inclui aba Simulador 2026)
├── dashboard.py             # Dashboard de fluxo e mapas
├── utils.py                 # Funções e lista dos municípios da RMC
├── NOTAS_MODELAGEM.md       # Descobertas de modelagem e investigação de features
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

> **Importante:** esta versão do pipeline também gera as variáveis de gravidade (`USOU_UTI`, `N_COMORBID`) em `gold_resp_ml.parquet`. Se você estiver usando uma base processada antiga (sem essas colunas), o `analise_permanencia_rmc.py` ainda roda, mas avisa e **não** treina o modelo com gravidade. Para ter o modelo explicativo completo, rode o pipeline desta versão.

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

### 4. Treinar o modelo do simulador 2026 (opcional)

Este passo treina o modelo de produção (previsão na admissão, **sem UTI**) e o salva em disco para o simulador do dashboard reutilizar:

```powershell
.\.venv\Scripts\python.exe .\previsao.py
```

O comando grava `modelo_permanencia_2026.joblib` em `TCC_Datasus/data/processed/` e imprime um exemplo de previsão. Não é obrigatório rodar à mão: se o modelo salvo não existir, a aba "Simulador 2026" do dashboard o treina sozinho na primeira vez.

### 5. Abrir os dashboards

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
- `modelo_permanencia_2026.joblib`: modelo de produção do simulador (gerado por `previsao.py`).

Em `resultados_permanencia/`:

- `metricas_modelos.csv`: métricas dos modelos no teste de 2025 (inclui o baseline por subgrupo e o HGBR com gravidade).
- `metricas_por_segmento_2025.csv`: desempenho dos modelos separado por segmento (urgência × eletiva).
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
