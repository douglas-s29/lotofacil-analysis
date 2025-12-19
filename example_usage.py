"""
Exemplo de uso da análise da Lotofácil
========================================

Este script demonstra como usar o módulo lotofacil_analysis
para realizar análises estatísticas e de Machine Learning.
"""

# Exemplo 1: Executar análise completa
print("="*60)
print("EXEMPLO 1: Análise Completa com Dados da API")
print("="*60)
print("""
Para executar a análise completa com dados reais da API:

    python lotofacil_analysis.py

Isso irá:
- Consultar a API da Lotofácil
- Processar os dados históricos
- Gerar análise de frequência
- Aplicar clustering KMeans
- Criar visualizações em PNG
""")

# Exemplo 2: Uso programático
print("\n" + "="*60)
print("EXEMPLO 2: Uso Programático das Funções")
print("="*60)
print("""
Você também pode importar e usar as funções individualmente:

from lotofacil_analysis import (
    fetch_lotofacil_data,
    structure_data_to_dataframe,
    calculate_number_frequency,
    plot_frequency_chart
)

# Obter dados
data = fetch_lotofacil_data()

# Estruturar em DataFrame
df = structure_data_to_dataframe(data)

# Calcular frequência
frequency = calculate_number_frequency(df)

# Gerar gráfico
plot_frequency_chart(frequency, 'meu_grafico.png')
""")

# Exemplo 3: Análise com dados customizados
print("\n" + "="*60)
print("EXEMPLO 3: Análise com Dados Customizados")
print("="*60)
print("""
Para testar com seus próprios dados:

import pandas as pd
from lotofacil_analysis import calculate_number_frequency, plot_frequency_chart

# Criar dados de exemplo
data = [
    {'concurso': 1, 'dezenas': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']},
    {'concurso': 2, 'dezenas': ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']},
    # ... mais concursos
]

df = pd.DataFrame(data)

# Calcular e visualizar
frequency = calculate_number_frequency(df)
plot_frequency_chart(frequency, 'custom_chart.png')
""")

# Exemplo 4: Ajustar número de clusters
print("\n" + "="*60)
print("EXEMPLO 4: Ajustar Número de Clusters")
print("="*60)
print("""
Para experimentar com diferentes números de clusters:

from lotofacil_analysis import (
    fetch_lotofacil_data,
    structure_data_to_dataframe,
    prepare_data_for_clustering,
    perform_kmeans_clustering,
    plot_cluster_visualization
)

data = fetch_lotofacil_data()
df = structure_data_to_dataframe(data)
features = prepare_data_for_clustering(df)

# Testar com 3 clusters
kmeans3, labels3, features_scaled, scaler3 = perform_kmeans_clustering(features, n_clusters=3)
plot_cluster_visualization(features_scaled, labels3, 'clusters_3.png')

# Testar com 7 clusters
kmeans7, labels7, features_scaled, scaler7 = perform_kmeans_clustering(features, n_clusters=7)
plot_cluster_visualization(features_scaled, labels7, 'clusters_7.png')
""")

# Exemplo 5: Gerar sugestão de jogo
print("\n" + "="*60)
print("EXEMPLO 5: Gerar Sugestão de Jogo Personalizada")
print("="*60)
print("""
Para gerar uma sugestão de jogo para o próximo concurso:

from lotofacil_analysis import (
    fetch_lotofacil_data,
    structure_data_to_dataframe,
    calculate_number_frequency,
    prepare_data_for_clustering,
    perform_kmeans_clustering,
    generate_game_suggestion,
    analyze_suggestion_statistics
)

# Obter e processar dados
data = fetch_lotofacil_data()
df = structure_data_to_dataframe(data)

# Calcular frequência e aplicar clustering
frequency = calculate_number_frequency(df)
features = prepare_data_for_clustering(df)
kmeans, labels, features_scaled, scaler = perform_kmeans_clustering(features)

# Gerar sugestão personalizada
suggestion = generate_game_suggestion(frequency, kmeans, features, labels)
print(f"Números sugeridos: {suggestion}")

# Analisar estatísticas da sugestão
analyze_suggestion_statistics(suggestion, frequency)

IMPORTANTE: Esta sugestão é apenas uma análise estatística educacional.
Loterias são completamente aleatórias e este programa NÃO garante
nenhum aumento nas chances de ganhar!
""")

print("\n" + "="*60)
print("Para mais informações, consulte o README.md")
print("="*60)
