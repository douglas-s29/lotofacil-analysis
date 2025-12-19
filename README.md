# Lotofácil Analysis - Machine Learning

Este projeto realiza análise estatística e de Machine Learning dos resultados da Lotofácil, utilizando dados obtidos diretamente da API oficial.

## 📋 Funcionalidades

1. **Consulta à API**: Consome dados da API da Lotofácil (https://loteriascaixa-api.herokuapp.com/api/lotofacil)
2. **Análise de Frequência**: Calcula e visualiza a frequência de cada número (1-25) no histórico de sorteios
3. **Clustering com KMeans**: Identifica padrões nos sorteios usando algoritmo de Machine Learning
4. **Sugestão de Jogo Personalizada**: Gera uma combinação de 15 números para o próximo concurso baseada em:
   - Frequências mais altas (números frequentes)
   - Clusters identificados (padrões de combinações)
   - Aleatoriedade para evitar previsibilidade
5. **Visualizações**: Gera gráficos profissionais para análise dos dados
6. **Análise Estatística**: Fornece estatísticas detalhadas sobre a sugestão gerada

## 🚀 Tecnologias Utilizadas

- **Python 3.x**
- **requests**: Para consumo da API
- **pandas**: Para estruturação e manipulação de dados
- **matplotlib**: Para visualização de dados
- **scikit-learn**: Para algoritmos de Machine Learning (KMeans, PCA)
- **numpy**: Para operações numéricas e geração de aleatoriedade

## 📦 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/douglas-s29/lotofacil-analysis.git
cd lotofacil-analysis
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🎯 Como Usar

Execute o script principal:
```bash
python lotofacil_analysis.py
```

O programa irá:
1. Consultar a API da Lotofácil
2. Processar os dados históricos
3. Gerar análise de frequência dos números
4. Aplicar clustering para identificar padrões
5. Criar visualizações (arquivos PNG)
6. Gerar uma sugestão personalizada de jogo para o próximo concurso
7. Apresentar estatísticas sobre a sugestão gerada

## 📊 Saídas Geradas

O programa gera os seguintes resultados:

### Visualizações (arquivos PNG):
- **frequency_chart.png**: Gráfico de barras mostrando a frequência de cada número sorteado
- **cluster_visualization.png**: Visualização 2D dos clusters identificados pelo KMeans

### Saída no console:
- **Análise de frequência**: Números mais e menos sorteados
- **Distribuição de clusters**: Quantos concursos em cada cluster
- **Padrões por cluster**: Números mais frequentes em cada grupo
- **Sugestão de jogo**: Combinação de 15 números para o próximo concurso
- **Estatísticas da sugestão**: Análise detalhada dos números sugeridos

## 🔍 Estrutura do Código

O código está organizado em funções bem documentadas:

- `fetch_lotofacil_data()`: Consulta a API e obtém os dados
- `structure_data_to_dataframe()`: Estrutura os dados em DataFrame
- `calculate_number_frequency()`: Calcula frequência dos números
- `plot_frequency_chart()`: Gera gráfico de frequência
- `prepare_data_for_clustering()`: Prepara dados para ML
- `perform_kmeans_clustering()`: Aplica algoritmo KMeans
- `plot_cluster_visualization()`: Visualiza os clusters
- `analyze_cluster_patterns()`: Analisa padrões encontrados
- `generate_game_suggestion()`: Gera sugestão personalizada de jogo
- `analyze_suggestion_statistics()`: Analisa estatísticas da sugestão

## 📝 Requisitos

Ver arquivo `requirements.txt` para lista completa de dependências.

## ⚠️ AVISO IMPORTANTE

**Este projeto é exclusivamente educacional e demonstra técnicas de análise de dados e Machine Learning.**

- Loterias são eventos com resultados **completamente aleatórios**
- Este programa **NÃO garante** nenhum aumento real nas chances de ganhar
- As sugestões geradas são apenas análises estatísticas sem valor preditivo
- Use apenas para fins de aprendizado e diversão
- Jogue com responsabilidade

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

Este projeto é de código aberto e está disponível para uso educacional e pessoal.