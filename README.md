# 🤖 Influência da IA no Trabalho
### Análise de Burnout, Produtividade e Risco de Atrito em Profissionais de Tecnologia

---

## 📌 Visão Geral

Este projeto analisa como a adoção de ferramentas de **Inteligência Artificial no ambiente de trabalho** impacta o bem-estar, a produtividade e a retenção de profissionais de tecnologia. A partir de um dataset com 21 variáveis coletadas em 2026, foram conduzidas duas frentes de trabalho:

1. **Análise Exploratória de Dados (EDA)** — investigação visual e estatística das relações entre uso de IA, burnout, satisfação e atrito.
2. **Modelagem Preditiva com Machine Learning** — construção de modelos de classificação e regressão para antecipar risco de atrito e nível de burnout.

---

## 📁 Estrutura do Projeto

```
📦 influencia-ia-no-trabalho/
├── ai_on_work.ipynb                   # Notebook principal (EDA + ML)
├── ai_worker_burnout_attrition_2026.csv # Dataset utilizado
└── README.md                            # Este arquivo
```

---

## 🗂️ Sobre o Dataset

**Arquivo:** `ai_worker_burnout_attrition_2026.csv`

O dataset contém informações sobre profissionais de tecnologia e sua relação com ferramentas de IA no trabalho. As principais variáveis são:

| Variável | Descrição |
|---|---|
| `job_role` | Cargo do profissional (Data Scientist, DevOps, AI Researcher, etc.) |
| `years_experience` | Anos de experiência na área |
| `education_level` | Nível de escolaridade (Self-taught, Bootcamp, Bachelor, Master, PhD) |
| `primary_ai_tool` | Principal ferramenta de IA utilizada |
| `ai_tools_used_per_day` | Quantidade de ferramentas de IA usadas por dia |
| `hours_with_ai_assistance_daily` | Horas diárias com assistência de IA |
| `ai_replaces_my_tasks_pct` | % de tarefas substituídas pela IA |
| `ai_adoption_stage` | Estágio de maturidade na adoção de IA |
| `weekly_ai_upskilling_hrs` | Horas semanais de aprendizado/upskilling em IA |
| `productivity_score` | Score de produtividade reportado |
| `burnout_score` | Score de esgotamento profissional |
| `job_satisfaction_1_5` | Satisfação com o trabalho (escala 1–5) |
| `fear_of_ai_replacement` | Medo de ser substituído pela IA (Low / Medium / High) |
| `attrition_risk` | Risco de atrito/desligamento (Low / Medium / High) |

---

## 🔬 Metodologia

### Parte I — Análise Exploratória

**1. Exploração e diagnóstico dos dados:** inspeção inicial com `head()`, `tail()`, `info()` e `describe()`, verificação de duplicatas e listagem de colunas.

**2. Tratamento de outliers:**
- *Outliers globais:* método IQR com fator `1.2` aplicado a `productivity_score`, `burnout_score`, `hours_with_ai_assistance_daily` e `ai_tools_used_per_day`.
- *Outliers por grupo:* remoção segmentada por `years_experience` na variável `burnout_score`.

**3. Análise visual (EDA):** 14 visualizações construídas com `matplotlib` e `seaborn` cobrindo distribuição de cargos, correlações, impacto da maturidade tecnológica, relações entre uso de IA, burnout e satisfação, e perfis multivariados de risco.

### Parte II — Modelagem Preditiva

**4. Preparação dos dados (`df_ml`):**
- Remoção do `employee_id`.
- `LabelEncoder` para transformar `attrition_risk` em `attrition_risk_encoded` (0/1/2).
- `pd.get_dummies(drop_first=True)` para One-Hot Encoding das variáveis categóricas (`job_role`, `education_level`, `country`, `industry`, `company_size`, `remote_work_type`, `primary_ai_tool`, `ai_adoption_stage`, `fear_of_ai_replacement`).

**5. Classificação — Risco de Atrito:**
- **Target:** `attrition_risk_encoded`
- **Modelos:** Random Forest Classifier e XGBoost Classifier
- **Divisão:** 80% treino / 20% teste (`random_state=42`)
- **Estratégia para desbalanceamento:** `class_weight='balanced'` (RF) e `compute_sample_weight` (XGBoost)
- **Métricas:** Matriz de Confusão e Classification Report (precision, recall, F1-score)

**6. Regressão — Previsão de Burnout:**
- **Target:** `burnout_score` (contínuo)
- **Modelos:** Random Forest Regressor e XGBoost Regressor
- **Divisão:** 80% treino / 20% teste (`random_state=42`)
- **Métricas:** MAE (Mean Absolute Error) e R² (Coeficiente de Determinação)

---

## 📊 Análises e Modelos Implementados

### Visualizações (EDA)

| # | Análise | Tipo de Gráfico |
|---|---|---|
| 1 | Mapa de correlação entre variáveis numéricas | Heatmap |
| 2 | Ferramentas de IA mais utilizadas | Countplot |
| 3 | Taxa de substituição de tarefas por cargo | Barplot |
| 4 | Distribuição de produtividade e horas de IA | Histplot + KDE |
| 5 | Distribuição de cargos no dataset | Countplot |
| 6 | Maturidade na IA vs. produtividade | Boxplot |
| 7 | Produtividade × Horas de IA (com burnout como cor) | Scatterplot |
| 8 | Horas de IA vs. Burnout (regressão) | Regplot |
| 9 | Substituição por IA vs. Satisfação (regressão) | Regplot |
| 10 | Upskilling semanal vs. Burnout por risco de atrito | Scatterplot |
| 11 | Burnout por anos de experiência | Boxplot |
| 12 | Risco de atrito vs. Burnout vs. Medo de IA | Barplot agrupado |
| 13 | Burnout por nível de escolaridade | Violin Plot |
| 14 | Horas de IA × Burnout × Atrito × Medo (multivariada) | Relplot / FacetGrid |

### Modelos de Machine Learning

| # | Tarefa | Algoritmo | Target | Avaliação |
|---|---|---|---|---|
| 1 | Classificação | Random Forest | `attrition_risk` | Matriz de Confusão + Classification Report |
| 2 | Classificação | XGBoost | `attrition_risk` | Matriz de Confusão + Classification Report |
| 3 | Regressão | Random Forest | `burnout_score` | MAE + R² |
| 4 | Regressão | XGBoost | `burnout_score` | MAE + R² |

---

## 💡 Principais Insights

### 🔥 Carga de Trabalho e Burnout
- Existe correlação positiva moderada entre **horas diárias com IA** e **burnout**: profissionais que utilizam IA por mais de 6 horas diárias tendem a reportar exaustão mais elevada.
- O **upskilling semanal** age como fator de pressão adicional — quanto mais horas investidas, maior o risco de atrito percebido.

### ⚠️ Risco de Atrito e Medo de Substituição
- O grupo de **Alto Risco de Atrito** apresenta a maior média de burnout (~74,0), especialmente quando combinado com medo de substituição médio ou alto.
- Profissionais com **baixo medo de substituição** correlacionam-se com maior satisfação no trabalho e menor burnout, independentemente do cargo.

### 🚀 Produtividade e Maturidade Tecnológica
- Empresas nos estágios **"Optimizing"** e **"AI-First"** apresentam scores de produtividade significativamente mais altos do que aquelas em "Experimenting".
- Cargos como **Data Scientist** e **AI Researcher** têm as maiores taxas de substituição de tarefas por IA, mas mantêm produtividade elevada — sugerindo adaptação bem-sucedida ao novo contexto.

### 🎓 Educação e Perfil Profissional
- O nível de escolaridade **não é preditor direto de burnout**: o estresse tecnológico afeta igualmente autodidatas e PhDs.
- O burnout é mais concentrado em profissionais com **5 a 12 anos de experiência** (*mid-career*).

---

## 🛠️ Tecnologias Utilizadas

| Categoria | Bibliotecas |
|---|---|
| Manipulação de dados | `pandas` |
| Visualização | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (RandomForest, LabelEncoder, train_test_split, métricas), `xgboost` |
| Ambiente | Python 3, Jupyter Notebook / Google Colab |

---

## ▶️ Como Executar

1. Clone o repositório ou baixe os arquivos.
2. Instale as dependências:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn xgboost jupyter
   ```
3. Coloque o arquivo `ai_worker_burnout_attrition_2026.csv` no mesmo diretório do notebook (ajuste o caminho no `pd.read_csv` se necessário).
4. Abra o notebook:
   ```bash
   jupyter notebook ai_on_work.ipynb
   ```

---

## 👤 Autor

Projeto desenvolvido como estudo analítico sobre os impactos da Inteligência Artificial no mercado de trabalho de tecnologia, combinando análise exploratória de dados e modelagem preditiva.
