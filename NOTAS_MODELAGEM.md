# Notas de modelagem — permanência hospitalar (RMC)

Registro das descobertas sobre o modelo de previsão do tempo de permanência
(`DIAS_PERM`), para aproveitar na escrita do TCC. Branch: `melhorias-modelagem-permanencia`.

Escopo: internações respiratórias (CID capítulo J) de residentes da RMC
**atendidos dentro da RMC**, sem óbitos. Treino 2022–2024, teste 2025.
Teste = 12.512 internações (11.782 urgências, 730 eletivas).

---

## 1. Desempenho global dos modelos (teste 2025)

| Modelo | MAE (dias) | RMSE (dias) | R² |
|---|---|---|---|
| Baseline: média global | 4,572 | 6,917 | −0,000 |
| Baseline: **média do subgrupo** | 4,236 | 6,634 | 0,080 |
| RandomForest sem município | 4,303 | 6,737 | 0,051 |
| RandomForest com município | 4,310 | 6,739 | 0,051 |
| HistGradientBoosting | 4,232 | 6,631 | 0,081 |
| HGBR (Poisson) | 4,230 | 6,630 | 0,081 |

**Leitura:** o melhor modelo explica só ~8% da variância. RMSE (6,6) ≫ MAE (4,2)
confirma a **cauda longa** (poucas internações muito longas dominam o erro
quadrático). Trocar de algoritmo (RF → HGBR → Poisson) quase não muda nada:
a diferença entre eles (~0,05 dia) é menor que o ruído entre subgrupos.

---

## 2. A métrica global engana — avaliação estratificada

Separando por tipo de atendimento (teste 2025):

| Segmento | n | Baseline subgrupo (MAE) | HGBR (MAE) | HGBR R² |
|---|---|---|---|---|
| Geral | 12.512 | 4,236 | 4,232 | 0,081 |
| **Urgência** | 11.782 | 4,348 | 4,341 | 0,062 |
| Eletiva | 730 | 2,438 | 2,464 | 0,181 |

**Descoberta central:** as eletivas são só **5,8%** dos casos, mas respondem por
**~51% da redução de erro** do modelo sobre o baseline global. São internações
de ~1 dia, triviais de prever ("eletiva = curto"). O "ganho" global do ML vinha
quase todo daí. No problema que importa (urgências longas), o ganho some.

Por grupo (doença × faixa × tipo), o HGBR é **pior que a média global** justamente
nos casos clinicamente relevantes: Pneumonia/Influenza de idoso e adulto urgente,
DPOC de idoso urgente — internações com mediana 5–6 dias e p90 de 16–20 dias.

---

## 3. O teste decisivo — baseline por subgrupo

Baseline por subgrupo = prever a **média de dias do próprio perfil**
(doença × faixa etária × tipo de atendimento), aprendida só no treino.
É uma tabela de lookup, sem ML.

| Segmento | Baseline subgrupo | HGBR | Ganho do ML |
|---|---|---|---|
| Geral | 4,236 | 4,232 | +0,004 |
| **Urgência** | 4,348 | 4,341 | **+0,007 (empate)** |

**Conclusão:** com as features atuais, o modelo de ML **não agrega nada** sobre
uma simples média estratificada. Em urgências é um empate técnico. A perda de
Poisson também não resolve. **O gargalo é informação, não algoritmo.**

> Implicação para o TCC: apresentar sempre a comparação contra o baseline de
> subgrupo, não contra a média global ingênua. Caso contrário, superestima-se o
> valor do modelo.

---

## 4. Que feature realmente falta? (investigação empírica)

As features atuais (`IDADE`, `FAIXA_ETARIA`, `ESTACAO`, `SUBCATEGORIA_CID`,
`TIPO_ATENDIMENTO`) são todas **administrativas/demográficas** — não capturam
**gravidade clínica**, que é o que determina internação longa.

Testei adicionar variáveis de gravidade dos parquets brutos do SIH (casadas por
`N_AIH`, 100% de correspondência). HGBR, mesmo split temporal:

| Conjunto de features | MAE Urgência | R² Urgência |
|---|---|---|
| Baseline subgrupo (referência) | 4,348 | 0,061 |
| **Atual** (sem gravidade) | 4,341 | 0,062 |
| + `N_COMORBID` | 4,363 | 0,054 |
| **+ `N_COMORBID` + `USOU_UTI`** | **3,952** | **0,209** |
| + `N_COMORBID` + `UTI_MES_TO` (vaza) | 3,374 | 0,427 |

**Respostas:**

- **`USOU_UTI` (passou pela UTI, sim/não) é a variável que faltava.** Sozinha
  derruba o MAE de urgências ~9% e **triplica o R²** (0,062 → 0,209). Só com ela
  o ML finalmente **ganha do baseline de subgrupo** (3,95 vs 4,35). 15,1% das
  internações usaram UTI (`UTI_MES_TO > 0`).
- **Contagem de comorbidade (`N_COMORBID`) sozinha não ajuda** — como contagem
  crua é fraca. Vale reformular (ex.: índice de Charlson/Elixhauser ponderado, ou
  comorbidades específicas) antes de descartar.
- **`UTI_MES_TO` (dias de UTI) não pode ser preditor**: dias de UTI são parte da
  própria permanência (quase circular). Entra aqui só como teto teórico.

### Ressalva de vazamento / tempo (importante)

`USOU_UTI` é conhecido **durante/após** a internação, não no momento da admissão.
Então ele serve para um modelo **explicativo** ("o que está associado a internações
longas") — e aí é válido e informativo — mas **não** para prever no instante da
admissão. O TCC precisa declarar qual é o objetivo. Variáveis de valor (`VAL_*`),
`QT_DIARIAS` e `UTI_MES_TO` são pós-desfecho e **não** devem ser preditores.

Variáveis do SIH ainda não exploradas que valem investigar como gravidade
legítima (conhecida próximo da admissão): `PROC_REA` (procedimento, p.ex.
ventilação mecânica), `COMPLEX` (quase constante aqui: 98% média complexidade,
fraca), diagnósticos secundários específicos.

---

## 5. Recomendações

1. **Adicionar `USOU_UTI` às features** — maior ganho possível, declarando que o
   modelo é explicativo (não de previsão na admissão). Exige reprocessar o
   pipeline para carregar `UTI_MES_TO`/`MARCA_UTI` dos brutos. Fazer em **branch
   separada** (mexe no `data_pipeline.py`), para não misturar com a avaliação honesta.
2. **Separar o problema**: modelar urgências e eletivas à parte. Eletiva de 1 dia
   não precisa de ML — pode ser regra.
3. **Reformular comorbidade**: testar Charlson/Elixhauser em vez de contagem crua.
4. **Sempre comparar com o baseline de subgrupo** e reportar MAE (robusto à cauda)
   como métrica principal, com avaliação estratificada.
5. **Reconsiderar o alvo**: como a cauda domina e o interesse de gestão é a
   internação longa, avaliar prever "permanência > p90" (classificação) em vez do
   número exato de dias.

---

## 6. Ressalva de causalidade

A diferença observado−esperado por hospital/município é **associação**, não causa.
Depois de ajustar pelo perfil do paciente, o município de atendimento praticamente
não melhora a previsão (4,310 vs 4,303) — ou seja, as diferenças entre hospitais
são pequenas. Rankings diretos de hospitais são enganosos; a leitura correta usa
razão observado/esperado com funnel plots (limites de controle por volume de casos).

---

---

## 7. Confirmação na branch `features-gravidade`

A feature foi implementada no pipeline (`USOU_UTI`, `N_COMORBID` em `gold_resp_ml`)
e o modelo `HGBR + gravidade (explicativo)` foi avaliado no mesmo framework honesto:

| Segmento | Baseline subgrupo | HGBR atual | HGBR + gravidade |
|---|---|---|---|
| Geral | 4,236 / R² 0,080 | 4,232 / 0,081 | **3,853 / 0,224** |
| Urgência | 4,348 / 0,061 | 4,341 / 0,062 | **3,952 / 0,209** |
| Eletiva | 2,438 / 0,154 | 2,464 / 0,181 | **2,248 / 0,277** |

Com gravidade, o ML finalmente **supera o baseline de subgrupo** em todos os
segmentos (em urgências: 3,95 vs 4,35; R² de 0,06 → 0,21). Confirma que o limitante
era informação clínica, não o algoritmo. Reforço da ressalva: `USOU_UTI` é
conhecido durante a internação → modelo **explicativo**, não de previsão na admissão.

---

*Gerado a partir de `resultados_permanencia/` (métricas, previsões e segmentos de
2025) e de experimento com os brutos do SIH casados por `N_AIH`.*
