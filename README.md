
# Modelo de Difusão Riemanniana para Estruturas de Proteínas

 *O que faz?* 
 
 Adapta o código do riemannian diffusion para os dados cujo modelo foldingdiff foi treinado.

*fonte de dados*: CATH Dataset

# Info. pertinente: 
O modelo foi treinado utilizando o dataset CATH (Class, Architecture, Topology, Homologous superfamily), uma classificação hierárquica de domínios de estruturas de proteínas.

Dataset Específico: cath-dataset-nonredundant-S40

Download: O dataset pode ser baixado do seguinte link:

cath-dataset-nonredundant-S40.pdb.tgz

# 🚀 Pré-requisitos e Setup

Para replicar o treinamento, siga os passos abaixo:

1. **Instalação de Dependências**

Antes de executar os scripts, é crucial instalar todas as bibliotecas Python necessárias. Elas estão listadas no arquivo requirements.txt.


pip install -r requirements.txt

2. **Baixe e Descompacte os Dados**

Faça o download do arquivo no link acima e descompacte-o.

3. **Organize os Arquivos PDB**

Crie um diretório chamado dompdb na raiz do projeto e mova todos os arquivos .pdb descompactados para dentro dele.

4. **Execute o Pré-processamento**

Rode o novo script de pré-processamento para converter os arquivos PDB em um formato de ângulos de torção.

*python preprocess_cath.py*

Este script irá ler todos os arquivos em dompdb, extrair os ângulos, criar segmentos de tamanho fixo (WINDOW_SIZE=50) e salvar o resultado em ./data/cath_s40_L64.tsv.

# 🧠 Inicie o Treinamento

Com os dados pré-processados, inicie o treinamento.


*python simple_impl/train.py*

# 🔧 Resumo das Modificações nos Arquivos

1. **preprocess_cath.py** (Novo Arquivo)

💡 Motivação: O projeto original não possuía uma maneira de processar os dados brutos de proteínas (.pdb). Era necessário um pipeline para extrair as informações relevantes (ângulos phi/psi) e formatá-las de uma maneira que o modelo pudesse consumir.

⚙️ *Funcionamento*: O script utiliza a biblioteca Bio.PDB para:

Iterar sobre todos os arquivos .pdb no diretório dompdb.

Extrair os ângulos de torção phi (φ) e psi (ψ) de cada polipeptídeo.

Aplicar uma lógica de janela deslizante para criar segmentos de ângulos de tamanho fixo (WINDOW_SIZE), garantindo que todas as entradas do modelo tenham a mesma dimensão.

Salvar todos os segmentos em um único arquivo .tsv para carregamento eficiente durante o treinamento. Pense que é uma lógica análoga ao que se faz numa série temporal.

2. **protein.py** (Modificado)

💡 *Motivação*: Para carregar os dados pré-processados pelo preprocess_cath.py, foi necessária uma classe Dataset personalizada do PyTorch.

✅ *Diferenças*:

A nova classe CATHDataset é projetada especificamente para ler o arquivo cath_s40_L64.tsv.

Ela define a geometria (manifold) do problema como um Torus, cuja dimensão é dinamicamente calculada com base no WINDOW_SIZE (dim = window_size * 2).

No método __getitem__, ela converte os ângulos (em radianos) para suas coordenadas no círculo trigonométrico (cosseno e seno), que é a representação que o modelo utiliza.

3. **train.py** (Modificado)

💡 Motivação: O script original era configurado para treinar com um dataset de RNA. Foi preciso adaptá-lo para carregar e treinar com os dados de proteínas do CATH.

✅ *Diferenças*:

Carregamento de Dados: Em vez de data.protein.RNA(...), o script agora instancia a nova CATHDataset:


dataset = CATHDataset(tsv_path="./data/cath_s40_L64.tsv", window_size=window_size)

Definição do Manifold: A geometria e a dimensão do toro não são mais fixas, mas sim obtidas diretamente do objeto dataset, tornando o código mais flexível.

Salvamento do Modelo: Ao final do treinamento, um código foi adicionado para salvar os pesos dos modelos treinados (modelf e modelb) em um arquivo trained_protein_model.pkl.

4. **losses.py** (Correção de Bug)

🐛 *Motivação*: A versão original gerava um AttributeError: 'numpy.ndarray' object has no attribute 'to'. A variável t (tempo de difusão) estava sendo criada como um array NumPy em vez de um tensor PyTorch.

✅ *Correção*: A linha que gera o tempo de amostragem t foi corrigida para garantir que t seja sempre um tensor PyTorch, permitindo operações de dispositivo (CPU/GPU) sem erros de tipo.


*Garante que 't' é um tensor PyTorch*
t = torch.rand(x.shape[0], device=x.device) * (mix.tf - eps) + eps
5. **solver.py** (Correção de Bug)

🐛 *Motivação*: Mesmo após corrigir losses.py, o AttributeError persistia. A causa raiz foi encontrada aqui: a função sampler recebia um tensor t mas o convertia internamente para um array NumPy ao usar np.linspace, recriando o problema.

✅ *Correção*: O array gerado pelo np.linspace é agora explicitamente convertido para um tensor PyTorch antes de tentar movê-lo para o dispositivo, resolvendo o bug de forma definitiva.

Original:

timesteps = np.linspace(mix.t0, ts.detach().cpu(), N)
timesteps = timesteps.to(x.device) # ERRO AQUI

Modificado:


timesteps_np = np.linspace(mix.t0, ts.detach().cpu().numpy(), N)
timesteps = torch.from_numpy(timesteps_np).float().to(x.device)



# Novo treinamento com UI do MLFLOW:
1. **simple_impl/mlflow_train.py** (Novo Arquivo)