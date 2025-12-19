"""
Lotofácil Analysis - Machine Learning Application
==================================================

This script consumes data from the Lotofácil API to perform statistical
and machine learning analysis on lottery draw results.

Features:
- Fetches historical lottery data from API
- Analyzes frequency of drawn numbers
- Applies KMeans clustering to identify patterns
- Generates visualizations for insights

Requirements: requests, pandas, matplotlib, scikit-learn
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from typing import Dict, List, Any, Tuple

# Constants
MAX_LOTTERY_NUMBER = 25


def fetch_lotofacil_data(api_url: str = "https://loteriascaixa-api.herokuapp.com/api/lotofacil") -> Dict[str, Any]:
    """
    Consume dados da API da Lotofácil.
    
    Esta função faz uma requisição HTTP GET para a API da Lotofácil e retorna
    os dados dos sorteios em formato JSON.
    
    Args:
        api_url (str): URL da API da Lotofácil
        
    Returns:
        Dict[str, Any]: Dados dos sorteios em formato de dicionário
        
    Raises:
        requests.RequestException: Se houver erro na requisição HTTP
    """
    try:
        print("Consultando API da Lotofácil...")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Dados obtidos com sucesso!")
        return data
    except requests.RequestException as e:
        print(f"✗ Erro ao consultar API: {e}")
        raise


def structure_data_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Estrutura os dados da API em um DataFrame do pandas.
    
    Esta função processa os dados brutos da API e organiza as informações
    relevantes (número do concurso e dezenas sorteadas) em um DataFrame
    para facilitar a análise.
    
    Args:
        data (Dict[str, Any]): Dados brutos obtidos da API
        
    Returns:
        pd.DataFrame: DataFrame com colunas 'concurso' e 'dezenas'
    """
    print("\nEstruturando dados em DataFrame...")
    
    # Extrair informações relevantes
    contests_data = []
    
    # Se os dados são uma lista de concursos
    if isinstance(data, list):
        for contest in data:
            contests_data.append({
                'concurso': contest.get('concurso'),
                'dezenas': contest.get('dezenas', [])
            })
    # Se os dados são um dicionário com o último concurso
    elif isinstance(data, dict):
        contests_data.append({
            'concurso': data.get('concurso'),
            'dezenas': data.get('dezenas', [])
        })
    
    df = pd.DataFrame(contests_data)
    print(f"✓ DataFrame criado com {len(df)} concursos")
    return df


def calculate_number_frequency(df: pd.DataFrame) -> pd.Series:
    """
    Calcula a frequência de cada número sorteado (1 a 25).
    
    Esta função analisa todos os sorteios e conta quantas vezes cada número
    entre 1 e 25 foi sorteado no histórico de concursos.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos concursos
        
    Returns:
        pd.Series: Série com a frequência de cada número (1-25)
    """
    print("\nCalculando frequência dos números...")
    
    # Inicializar contador para números de 1 a MAX_LOTTERY_NUMBER
    frequency = {str(i).zfill(2): 0 for i in range(1, MAX_LOTTERY_NUMBER + 1)}
    
    # Contar frequência de cada número
    for dezenas in df['dezenas']:
        if isinstance(dezenas, list):
            for numero in dezenas:
                if numero in frequency:
                    frequency[numero] += 1
    
    frequency_series = pd.Series(frequency).sort_index()
    print(f"✓ Frequência calculada para {len(frequency_series)} números")
    return frequency_series


def plot_frequency_chart(frequency: pd.Series, output_file: str = "frequency_chart.png"):
    """
    Plota um gráfico de barras mostrando a frequência de cada número.
    
    Esta função cria uma visualização da frequência com que cada número
    (1 a 25) foi sorteado no histórico de concursos da Lotofácil.
    
    Args:
        frequency (pd.Series): Série com a frequência de cada número
        output_file (str): Nome do arquivo para salvar o gráfico
    """
    print("\nGerando gráfico de frequência...")
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(frequency)), frequency.values, color='steelblue', edgecolor='black')
    
    # Adicionar labels e título
    plt.xlabel('Números', fontsize=12, fontweight='bold')
    plt.ylabel('Frequência', fontsize=12, fontweight='bold')
    plt.title('Frequência de Números Sorteados na Lotofácil', fontsize=14, fontweight='bold')
    plt.xticks(range(len(frequency)), frequency.index, rotation=45)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adicionar valores no topo das barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo em: {output_file}")
    plt.close()


def prepare_data_for_clustering(df: pd.DataFrame) -> np.ndarray:
    """
    Prepara os dados para clustering criando uma matriz de features.
    
    Esta função transforma os dados dos sorteios em uma matriz binária
    onde cada linha representa um concurso e cada coluna representa
    se um número específico (1-25) foi sorteado (1) ou não (0).
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos concursos
        
    Returns:
        np.ndarray: Matriz de features para clustering
    """
    print("\nPreparando dados para clustering...")
    
    # Criar matriz de features (concursos x números)
    features = []
    
    for dezenas in df['dezenas']:
        # Criar vetor binário: 1 se número foi sorteado, 0 caso contrário
        vector = [0] * MAX_LOTTERY_NUMBER
        if isinstance(dezenas, list):
            for numero in dezenas:
                numero_int = int(numero) - 1  # Converter para índice (0-24)
                if 0 <= numero_int < MAX_LOTTERY_NUMBER:
                    vector[numero_int] = 1
        features.append(vector)
    
    features_array = np.array(features)
    print(f"✓ Matriz de features criada: {features_array.shape}")
    return features_array


def perform_kmeans_clustering(features: np.ndarray, n_clusters: int = 5) -> Tuple[KMeans, np.ndarray, np.ndarray, StandardScaler]:
    """
    Aplica o algoritmo KMeans para identificar padrões nos sorteios.
    
    Esta função utiliza o algoritmo KMeans para agrupar concursos com
    base nos números sorteados, identificando padrões de combinações
    frequentes.
    
    Args:
        features (np.ndarray): Matriz de features dos concursos
        n_clusters (int): Número de clusters a serem criados
        
    Returns:
        Tuple: (modelo KMeans, labels dos clusters, features normalizadas, scaler)
    """
    print(f"\nAplicando KMeans clustering com {n_clusters} clusters...")
    
    # Normalizar os dados
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    print(f"✓ Clustering concluído!")
    print(f"  Inércia: {kmeans.inertia_:.2f}")
    
    # Mostrar distribuição dos clusters
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Distribuição dos clusters:")
    for cluster_id, count in zip(unique, counts):
        print(f"    Cluster {cluster_id}: {count} concursos ({count/len(labels)*100:.1f}%)")
    
    return kmeans, labels, features_scaled, scaler


def plot_cluster_visualization(features_scaled: np.ndarray, labels: np.ndarray, 
                                output_file: str = "cluster_visualization.png"):
    """
    Visualiza os clusters criados pelo KMeans.
    
    Esta função cria uma visualização 2D dos clusters utilizando as duas
    primeiras componentes principais dos dados para redução de dimensionalidade.
    
    Args:
        features_scaled (np.ndarray): Features normalizadas
        labels (np.ndarray): Labels dos clusters
        output_file (str): Nome do arquivo para salvar o gráfico
    """
    print("\nGerando visualização dos clusters...")
    
    # Reduzir dimensionalidade para visualização 2D usando PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='black', s=50)
    
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
               fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
               fontsize=12, fontweight='bold')
    plt.title('Visualização dos Clusters de Sorteios (KMeans + PCA)', 
              fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualização salva em: {output_file}")
    plt.close()


def analyze_cluster_patterns(df: pd.DataFrame, labels: np.ndarray, features: np.ndarray):
    """
    Analisa os padrões encontrados em cada cluster.
    
    Esta função examina cada cluster identificado pelo KMeans e mostra
    quais números são mais frequentes em cada grupo de sorteios.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados dos concursos
        labels (np.ndarray): Labels dos clusters
        features (np.ndarray): Matriz de features dos concursos
    """
    print("\n" + "="*60)
    print("ANÁLISE DOS PADRÕES POR CLUSTER")
    print("="*60)
    
    n_clusters = len(np.unique(labels))
    
    for cluster_id in range(n_clusters):
        # Selecionar concursos do cluster
        cluster_mask = labels == cluster_id
        cluster_features = features[cluster_mask]
        
        # Calcular frequência média de cada número no cluster
        avg_frequency = cluster_features.mean(axis=0)
        
        # Identificar números mais frequentes (acima da média)
        frequent_numbers = [i+1 for i, freq in enumerate(avg_frequency) if freq > 0.5]
        
        print(f"\n📊 CLUSTER {cluster_id}")
        print(f"   Número de concursos: {cluster_mask.sum()}")
        print(f"   Números mais frequentes: {frequent_numbers}")
        print(f"   Frequência média: {avg_frequency.mean():.2f}")


def main():
    """
    Função principal que executa todo o pipeline de análise.
    
    Esta função orquestra todas as etapas do processo:
    1. Consulta à API da Lotofácil
    2. Estruturação dos dados em DataFrame
    3. Análise de frequência dos números
    4. Visualização da frequência
    5. Clustering com KMeans
    6. Visualização dos clusters
    7. Análise dos padrões encontrados
    """
    print("="*60)
    print("ANÁLISE DA LOTOFÁCIL COM MACHINE LEARNING")
    print("="*60)
    
    try:
        # 1. Consultar API
        data = fetch_lotofacil_data()
        
        # 2. Estruturar dados em DataFrame
        df = structure_data_to_dataframe(data)
        
        # 3. Calcular frequência dos números
        frequency = calculate_number_frequency(df)
        
        # 4. Plotar gráfico de frequência
        plot_frequency_chart(frequency)
        
        # 5. Preparar dados para clustering
        features = prepare_data_for_clustering(df)
        
        # 6. Aplicar KMeans clustering
        kmeans, labels, features_scaled, scaler = perform_kmeans_clustering(features, n_clusters=5)
        
        # 7. Visualizar clusters
        plot_cluster_visualization(features_scaled, labels)
        
        # 8. Analisar padrões dos clusters
        analyze_cluster_patterns(df, labels, features)
        
        print("\n" + "="*60)
        print("✓ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("="*60)
        print("\nArquivos gerados:")
        print("  - frequency_chart.png")
        print("  - cluster_visualization.png")
        
    except Exception as e:
        print(f"\n✗ Erro durante a execução: {e}")
        raise


if __name__ == "__main__":
    main()
