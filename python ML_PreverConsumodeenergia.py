"""
ANÁLISE DE SUSTENTABILIDADE E ENERGIA COM MACHINE LEARNING
--------------------------------------------------------

INSTRUÇÕES DE INSTALAÇÃO:
Antes de executar este script, instale as bibliotecas necessárias executando:
    pip install pandas numpy matplotlib scikit-learn

Se o comando acima não funcionar, tente:
    python -m pip install pandas numpy matplotlib scikit-learn
"""

#pip install scikit-learn pandas numpy matplotlib seaborn - Intalar no CMD 
#pip install -U scikit-learn
#pip install -U matplotlib
#pip install -U seaborn
#pip install -U pandas
#pip install -U numpy
#pip install -U openpyxl
#pip install pandas numpy matplotlib scikit-learn
#pip install openpyxl


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Criar diretório para salvar as imagens
if not os.path.exists('resultados'):
    os.makedirs('resultados')

# Definir uma seed para reprodutibilidade
np.random.seed(42)

print("Gerando dados para análise de consumo de energia sustentável...")

# Gerar dados sintéticos para demonstração
def generate_energy_data(n_samples=1000):
    """
    Gera dados sintéticos para análise de consumo de energia sustentável.
    """
    # Características dos edifícios
    building_size = np.random.normal(5000, 2000, n_samples)  # Tamanho em m²
    building_age = np.random.randint(1, 50, n_samples)  # Idade em anos
    
    # Características de energia renovável
    solar_panels = np.random.randint(0, 30, n_samples)  # Número de painéis solares
    wind_turbines = np.random.randint(0, 5, n_samples)  # Número de turbinas eólicas
    
    # Características de isolamento e eficiência (simplificadas para valores numéricos)
    insulation_quality = np.random.randint(1, 4, n_samples)  # 1=baixa, 2=média, 3=alta
    window_efficiency = np.random.randint(1, 4, n_samples)  # 1=simples, 2=duplo, 3=triplo
    smart_systems = np.random.choice([0, 1], n_samples)  # Sistemas inteligentes (0=não, 1=sim)
    
    # Características climáticas
    avg_temperature = np.random.normal(18, 8, n_samples)  # Temperatura média anual em °C
    precipitation = np.random.normal(800, 300, n_samples)  # Precipitação anual em mm
    
    # Práticas sustentáveis
    water_recycling = np.random.choice([0, 1], n_samples)  # Sistema de reciclagem de água
    green_roof = np.random.choice([0, 1], n_samples)  # Telhado verde
    
    # Calcular consumo de energia (kWh/ano) com base nas características
    base_consumption = building_size * 0.1  # Base: 0.1 kWh por m² por ano
    
    # Ajustes para características do edifício
    age_factor = building_age * 50  # Edifícios mais antigos consomem mais
    
    # Ajustes para energia renovável
    solar_saving = solar_panels * 500  # Cada painel economiza 500 kWh/ano
    wind_saving = wind_turbines * 2000  # Cada turbina economiza 2000 kWh/ano
    
    # Ajustes para isolamento e eficiência
    insulation_factor = np.choose(insulation_quality - 1, [1.3, 1.0, 0.7])
    window_factor = np.choose(window_efficiency - 1, [1.2, 1.0, 0.8])
    smart_saving = smart_systems * 1000  # Sistemas inteligentes economizam 1000 kWh/ano
    
    # Ajustes para clima
    temperature_factor = 100 * (abs(avg_temperature - 18) / 10)
    
    # Ajustes para práticas sustentáveis
    water_saving = water_recycling * 500  # Reciclagem de água economiza energia indiretamente
    green_roof_saving = green_roof * 800  # Telhado verde melhora isolamento
    
    # Cálculo do consumo final com ruído aleatório
    energy_consumption = (
        base_consumption + 
        age_factor - 
        solar_saving - 
        wind_saving + 
        temperature_factor -
        water_saving -
        green_roof_saving
    ) * insulation_factor * window_factor
    
    # Subtrair economia de sistemas inteligentes
    energy_consumption -= smart_saving
    
    # Adicionar ruído aleatório (±10%)
    noise = np.random.normal(0, 0.1 * abs(energy_consumption))
    energy_consumption += noise
    
    # Garantir que o consumo seja sempre positivo
    energy_consumption = np.maximum(energy_consumption, 0)
    
    # Converter valores numéricos para categorias (para manter o mesmo formato dos dados)
    insulation_mapping = {1: 'baixa', 2: 'média', 3: 'alta'}
    window_mapping = {1: 'simples', 2: 'duplo', 3: 'triplo'}
    
    insulation_quality_cat = [insulation_mapping[i] for i in insulation_quality]
    window_efficiency_cat = [window_mapping[i] for i in window_efficiency]
    
    # Criar DataFrame
    data = pd.DataFrame({
        'tamanho_edificio': building_size,
        'idade_edificio': building_age,
        'paineis_solares': solar_panels,
        'turbinas_eolicas': wind_turbines,
        'qualidade_isolamento': insulation_quality_cat,
        'eficiencia_janelas': window_efficiency_cat,
        'sistemas_inteligentes': smart_systems,
        'temperatura_media': avg_temperature,
        'precipitacao': precipitation,
        'reciclagem_agua': water_recycling,
        'telhado_verde': green_roof,
        'consumo_energia': energy_consumption
    })
    
    return data

# Gerar dados
print("Gerando conjunto de dados para análise...")
data = generate_energy_data(1000)

# Mostrar as primeiras linhas dos dados
print("\nPrimeiras linhas do conjunto de dados:")
print(data.head())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(data.describe())

# Previsão do consumo de energia
print("\nPreparando modelo de previsão de consumo de energia...")

# Preparação simplificada para modelagem (sem pipeline completo)
# Para variáveis categóricas, vamos usar codificação manual simples
data['qualidade_isolamento_num'] = data['qualidade_isolamento'].map({'baixa': 1, 'média': 2, 'alta': 3})
data['eficiencia_janelas_num'] = data['eficiencia_janelas'].map({'simples': 1, 'duplo': 2, 'triplo': 3})

# Selecionar apenas as características numéricas para o modelo
features = [
    'tamanho_edificio', 'idade_edificio', 'paineis_solares', 'turbinas_eolicas',
    'qualidade_isolamento_num', 'eficiencia_janelas_num', 'sistemas_inteligentes',
    'temperatura_media', 'precipitacao', 'reciclagem_agua', 'telhado_verde'
]

X = data[features]
y = data['consumo_energia']

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Random Forest simplificado
print("Treinando modelo Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nResultados da Avaliação do Modelo:")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Análise de importância das características
print("\nImportância das Características:")
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

for i, idx in enumerate(sorted_idx):
    print(f"{features[idx]}: {feature_importance[idx]:.4f}")

# Visualizar importância das características
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.xlabel('Importância Relativa')
plt.title('Importância das Características para o Consumo de Energia')
plt.tight_layout()
plt.savefig('resultados/feature_importance.png')
print("\nGráfico de importância das características salvo em 'resultados/feature_importance.png'")

# Visualizar predições vs. valores reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Consumo Real (kWh/ano)')
plt.ylabel('Consumo Previsto (kWh/ano)')
plt.title('Valores Reais vs. Previstos')
plt.tight_layout()
plt.savefig('resultados/predictions_vs_actual.png')
print("Gráfico de previsões vs. valores reais salvo em 'resultados/predictions_vs_actual.png'")

# Função para fazer previsões com novos dados
def predict_energy_consumption(model, building_size, building_age, solar_panels, wind_turbines,
                             insulation_quality, window_efficiency, smart_systems,
                             avg_temperature, precipitation, water_recycling, green_roof):
    """
    Faz previsões de consumo de energia para novos dados.
    """
    # Converter variáveis categóricas para numéricas
    insulation_map = {'baixa': 1, 'média': 2, 'alta': 3}
    window_map = {'simples': 1, 'duplo': 2, 'triplo': 3}
    
    insulation_num = insulation_map.get(insulation_quality, 2)  # padrão é 'média'
    window_num = window_map.get(window_efficiency, 2)  # padrão é 'duplo'
    
    # Criar DataFrame com um exemplo
    example = pd.DataFrame({
        'tamanho_edificio': [building_size],
        'idade_edificio': [building_age],
        'paineis_solares': [solar_panels],
        'turbinas_eolicas': [wind_turbines],
        'qualidade_isolamento_num': [insulation_num],
        'eficiencia_janelas_num': [window_num],
        'sistemas_inteligentes': [smart_systems],
        'temperatura_media': [avg_temperature],
        'precipitacao': [precipitation],
        'reciclagem_agua': [water_recycling],
        'telhado_verde': [green_roof]
    })
    
    # Fazer a previsão
    prediction = model.predict(example)[0]
    return prediction

# Exemplos para previsão
print("\n--- Exemplos de Previsão de Consumo de Energia ---")

# Exemplo 1: Edifício moderno com tecnologias sustentáveis
exemplo1 = predict_energy_consumption(
    model, 
    building_size=3000, 
    building_age=5, 
    solar_panels=20, 
    wind_turbines=0,
    insulation_quality='alta', 
    window_efficiency='triplo', 
    smart_systems=1,
    avg_temperature=22, 
    precipitation=500, 
    water_recycling=1, 
    green_roof=1
)
print(f"Exemplo 1 - Edifício moderno sustentável: {exemplo1:.2f} kWh/ano")

# Exemplo 2: Edifício antigo sem tecnologias sustentáveis
exemplo2 = predict_energy_consumption(
    model, 
    building_size=8000, 
    building_age=30, 
    solar_panels=0, 
    wind_turbines=0,
    insulation_quality='baixa', 
    window_efficiency='simples', 
    smart_systems=0,
    avg_temperature=15, 
    precipitation=1000, 
    water_recycling=0, 
    green_roof=0
)
print(f"Exemplo 2 - Edifício antigo sem sustentabilidade: {exemplo2:.2f} kWh/ano")

# Exemplo 3: Edifício de meia-idade com algumas tecnologias sustentáveis
exemplo3 = predict_energy_consumption(
    model, 
    building_size=5500, 
    building_age=15, 
    solar_panels=10, 
    wind_turbines=2,
    insulation_quality='média', 
    window_efficiency='duplo', 
    smart_systems=1,
    avg_temperature=18, 
    precipitation=800, 
    water_recycling=1, 
    green_roof=0
)
print(f"Exemplo 3 - Edifício de meia-idade com algumas tecnologias: {exemplo3:.2f} kWh/ano")

# Simulação de otimização para o edifício antigo
print("\n--- Simulação de Otimizações para o Edifício Antigo ---")
print(f"Consumo original do edifício antigo: {exemplo2:.2f} kWh/ano")

# Simulação 1: Adicionar painéis solares
sim1 = predict_energy_consumption(
    model, 
    building_size=8000, 
    building_age=30, 
    solar_panels=25,  # Adicionando painéis solares
    wind_turbines=0,
    insulation_quality='baixa', 
    window_efficiency='simples', 
    smart_systems=0,
    avg_temperature=15, 
    precipitation=1000, 
    water_recycling=0, 
    green_roof=0
)
economia1 = exemplo2 - sim1
print(f"Adição de painéis solares: {sim1:.2f} kWh/ano (Economia: {economia1:.2f} kWh/ano, {(economia1/exemplo2*100):.1f}%)")

# Simulação 2: Melhorar isolamento
sim2 = predict_energy_consumption(
    model, 
    building_size=8000, 
    building_age=30, 
    solar_panels=0, 
    wind_turbines=0,
    insulation_quality='alta',  # Melhorando isolamento
    window_efficiency='simples', 
    smart_systems=0,
    avg_temperature=15, 
    precipitation=1000, 
    water_recycling=0, 
    green_roof=0
)
economia2 = exemplo2 - sim2
print(f"Melhoria do isolamento: {sim2:.2f} kWh/ano (Economia: {economia2:.2f} kWh/ano, {(economia2/exemplo2*100):.1f}%)")

# Simulação 3: Adicionar sistemas inteligentes
sim3 = predict_energy_consumption(
    model, 
    building_size=8000, 
    building_age=30, 
    solar_panels=0, 
    wind_turbines=0,
    insulation_quality='baixa', 
    window_efficiency='simples', 
    smart_systems=1,  # Adicionando sistemas inteligentes
    avg_temperature=15, 
    precipitation=1000, 
    water_recycling=0, 
    green_roof=0
)
economia3 = exemplo2 - sim3
print(f"Implementação de sistemas inteligentes: {sim3:.2f} kWh/ano (Economia: {economia3:.2f} kWh/ano, {(economia3/exemplo2*100):.1f}%)")

# Simulação 4: Todas as melhorias
sim4 = predict_energy_consumption(
    model, 
    building_size=8000, 
    building_age=30, 
    solar_panels=25, 
    wind_turbines=1,
    insulation_quality='alta', 
    window_efficiency='triplo', 
    smart_systems=1,
    avg_temperature=15, 
    precipitation=1000, 
    water_recycling=1, 
    green_roof=1
)
economia4 = exemplo2 - sim4
print(f"Todas as melhorias combinadas: {sim4:.2f} kWh/ano (Economia: {economia4:.2f} kWh/ano, {(economia4/exemplo2*100):.1f}%)")

# Visualizar resultados das simulações
resultados = {
    'Adição de painéis solares': economia1/exemplo2*100,
    'Melhoria do isolamento': economia2/exemplo2*100,
    'Sistemas inteligentes': economia3/exemplo2*100,
    'Todas as melhorias': economia4/exemplo2*100
}

plt.figure(figsize=(10, 6))
plt.bar(resultados.keys(), resultados.values())
plt.ylabel('Economia de Energia (%)')
plt.title('Economia de Energia por Intervenção Sustentável')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('resultados/optimization_results.png')
print("\nGráfico de resultados de otimização salvo em 'resultados/optimization_results.png'")

print("\nAnálise de sustentabilidade e energia concluída com sucesso!")