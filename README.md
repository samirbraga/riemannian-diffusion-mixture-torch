### 🚀 Começando
Siga os passos abaixo para configurar o ambiente e replicar o treinamento.

#### 1. Instale as Dependências
Clone o repositório e instale todas as bibliotecas Python necessárias a partir do arquivo requirements.txt.

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_DIRETORIO>
pip install -r requirements.txt
```

#### 2. Prepare os Dados
Faça o download do dataset no link acima e organize os arquivos.

**Instruções:**
1. Descompacte o arquivo cath-dataset-nonredundant-S40.pdb.tgz.  
2. Crie um diretório chamado dompdb na raiz do projeto.  
3. Mova todos os arquivos .pdb descompactados para dentro da pasta dompdb.

#### 3. Execute o Pré-processamento
Rode o script para converter os arquivos PDB em um formato de ângulos de torção, que o modelo pode utilizar.

```bash
python preprocess_cath.py
```

Este script irá gerar o arquivo ./data/cath_s40_L64.tsv, que contém os dados de treinamento.

---

### 🧠 Treinamento
Com os dados prontos, você pode iniciar o treinamento do modelo.

#### Treinamento Padrão
Execute o script de treino principal. Ao final, os pesos do modelo serão salvos em trained_protein_model.pkl.

```bash
python simple_impl/train.py
```

#### Treinamento com Tracking (MLflow)
Para uma visualização avançada das métricas e um melhor gerenciamento de experimentos, utilize a versão com MLflow.

```bash
python simple_impl/mlflow_train.py
```

Para visualizar os resultados, execute em outro terminal:

```bash
mlflow ui
```


---

### 🔧 Resumo das Modificações no Código
Aqui estão os detalhes das principais alterações e correções feitas no projeto original.

#### 1. preprocess_cath.py (Novo Arquivo)
💡 Motivação: O projeto original não tinha um pipeline para processar dados brutos de proteínas (.pdb).  
⚙️ Funcionamento:
- Utiliza Bio.PDB para iterar sobre os arquivos na pasta dompdb.
- Extrai os ângulos de torção phi (φ) e psi (ψ).
- Aplica uma janela deslizante (WINDOW_SIZE=50) para criar segmentos de tamanho fixo, similar a uma série temporal.
- Salva todos os segmentos em um único arquivo .tsv para carregamento eficiente.

#### 2. data/protein.py (Modificado)
💡 Motivação: Criar uma classe Dataset personalizada do PyTorch para carregar os dados pré-processados.  
✅ Diferenças:
- A nova classe CATHDataset lê o arquivo .tsv gerado.
- Define a geometria (manifold) como um Torus, com dimensão calculada dinamicamente (dim = window_size * 2).
- Converte os ângulos (radianos) em coordenadas (cos, sin) no método __getitem__, que é a representação usada pelo modelo.

#### 3. simple_impl/train.py (Modificado)
💡 Motivação: Adaptar o script, antes focado em RNA, para treinar com os dados de proteínas do CATH.  
✅ Diferenças:
- Carregamento de Dados: Instancia a nova CATHDataset.
- Flexibilidade: A geometria (manifold) e a dimensão são obtidas diretamente do objeto dataset, tornando o código mais robusto.
- Salvamento do Modelo: Adicionado código para salvar os pesos do modelo em trained_protein_model.pkl ao final do treino.

#### 4. losses.py (Correção de Bug)
🐛 O Bug: Ocorria o erro AttributeError: 'numpy.ndarray' object has no attribute 'to', pois a variável de tempo t era um array NumPy em vez de um tensor PyTorch.  
✅ A Correção: A linha foi modificada para garantir que t seja sempre um tensor PyTorch, permitindo operações de dispositivo (CPU/GPU).

```python
# Garante que 't' é um tensor PyTorch
t = torch.rand(x.shape[0], device=x.device) * (mix.tf - eps) + eps
```

#### 5. solver.py (Correção de Bug)
🐛 O Bug: O mesmo AttributeError persistia, pois a função sampler convertia internamente o tensor t de volta para NumPy ao usar np.linspace.  
✅ A Correção: O array gerado por np.linspace agora é explicitamente reconvertido para um tensor PyTorch antes de ser movido para o dispositivo.

```python
# Antes (com erro)
# timesteps = np.linspace(mix.t0, ts.detach().cpu(), N)
# timesteps = timesteps.to(x.device) 

# Depois (corrigido)
timesteps_np = np.linspace(mix.t0, ts.detach().cpu().numpy(), N)

ps: adicionado treinamento com mlflow para tracking de parâmetros em simple_impl/mlflow_train.py
timesteps = torch.from_numpy(timesteps_np).float().to(x.device)
```
