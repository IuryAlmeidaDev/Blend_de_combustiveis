import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==============================
# Configuração para labels em português
# ==============================
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

# ==============================
# Hiperparâmetros do Algoritmo Genético
# ==============================
TAMANHO_POPULACAO = 100        # Número de indivíduos por geração
TAXA_CROSSOVER = 0.7           # Probabilidade de crossover
TAXA_MUTACAO = 0.01            # Probabilidade de mutação
NUM_GERACOES = 100             # Número de gerações
TORNEIO_K = 3                  # Número de indivíduos no torneio de seleção

# ==============================
# Dados dos componentes do combustível
# ==============================
class ComponenteDados:
    def __init__(self):
        self.custos = np.array([2.50, 3.00, 1.80, 2.80])  # R$/L
        self.octanagem = np.array([95, 88, 85, 92])       # RON
        self.pressao_vapor = np.array([65, 55, 58, 64])  # kPa
        self.teor_benzeno = np.array([2.5, 1.5, 1.2, 2.0]) # % v/v
        self.teor_enxofre = np.array([60, 40, 35, 55])    # ppm

# ==============================
# Especificações da mistura
# ==============================
class Especificacoes:
    def __init__(self):
        self.octanagem_min = 91        # RON
        self.pressao_vapor_max = 62    # kPa
        self.benzeno_max = 2.0         # % v/v
        self.enxofre_max = 50          # ppm

# ==============================
# Algoritmo Genético
# ==============================
class AlgoritmoGenetico:
    def __init__(self, tamanho_pop=TAMANHO_POPULACAO, taxa_crossover=TAXA_CROSSOVER,
                 taxa_mutacao=TAXA_MUTACAO, geracoes=NUM_GERACOES, torneio_k=TORNEIO_K):
        self.tamanho_pop = tamanho_pop
        self.taxa_crossover = taxa_crossover
        self.taxa_mutacao = taxa_mutacao
        self.geracoes = geracoes
        self.torneio_k = torneio_k
        self.componentes = ComponenteDados()
        self.specs = Especificacoes()
        self.historico_custos = []
        self.melhor_solucao = None
        self.melhor_custo = float('inf')
    
    # ---------- Criação de indivíduos e população ----------
    def criar_individuo(self):
        proporcoes = np.random.random(4)
        return proporcoes / proporcoes.sum()
    
    def criar_populacao(self):
        return np.array([self.criar_individuo() for _ in range(self.tamanho_pop)])
    
    # ---------- Cálculo de propriedades e fitness ----------
    def calcular_propriedades(self, individuo):
        octanagem = np.dot(individuo, self.componentes.octanagem)
        pressao = np.dot(individuo, self.componentes.pressao_vapor)
        benzeno = np.dot(individuo, self.componentes.teor_benzeno)
        enxofre = np.dot(individuo, self.componentes.teor_enxofre)
        return octanagem, pressao, benzeno, enxofre
    
    def calcular_custo(self, individuo):
        return np.dot(individuo, self.componentes.custos)
    
    def calcular_penalidades(self, individuo):
        oct, pres, benz, enx = self.calcular_propriedades(individuo)
        penalidade = 0
        if oct < self.specs.octanagem_min:
            penalidade += 100 * (self.specs.octanagem_min - oct)
        if pres > self.specs.pressao_vapor_max:
            penalidade += 50 * (pres - self.specs.pressao_vapor_max)
        if benz > self.specs.benzeno_max:
            penalidade += 200 * (benz - self.specs.benzeno_max)
        if enx > self.specs.enxofre_max:
            penalidade += 150 * (enx - self.specs.enxofre_max)
        return penalidade
    
    def fitness(self, individuo):
        return self.calcular_custo(individuo) + self.calcular_penalidades(individuo)
    
    # ---------- Operadores genéticos ----------
    def selecao_torneio(self, populacao, fitness_values):
        idx = np.random.choice(len(populacao), self.torneio_k)
        melhor_idx = idx[np.argmin(fitness_values[idx])]
        return populacao[melhor_idx].copy()
    
    def crossover(self, pai1, pai2):
        if np.random.random() < self.taxa_crossover:
            mascara = np.random.random(4) < 0.5
            filho = np.where(mascara, pai1, pai2)
            return filho / filho.sum()
        return pai1.copy()
    
    def mutacao(self, individuo):
        if np.random.random() < self.taxa_mutacao:
            idx = np.random.randint(4)
            individuo[idx] += np.random.normal(0, 0.1)
            individuo = np.abs(individuo)
            individuo = individuo / individuo.sum()
        return individuo
    
    # ---------- Execução do AG ----------
    def executar(self):
        populacao = self.criar_populacao()
        for geracao in range(self.geracoes):
            fitness_values = np.array([self.fitness(ind) for ind in populacao])
            melhor_idx = np.argmin(fitness_values)
            melhor_fitness = fitness_values[melhor_idx]
            self.historico_custos.append(melhor_fitness)
            if melhor_fitness < self.melhor_custo:
                self.melhor_custo = melhor_fitness
                self.melhor_solucao = populacao[melhor_idx].copy()
            nova_populacao = [populacao[melhor_idx].copy()]  # Elitismo
            while len(nova_populacao) < self.tamanho_pop:
                pai1 = self.selecao_torneio(populacao, fitness_values)
                pai2 = self.selecao_torneio(populacao, fitness_values)
                filho = self.crossover(pai1, pai2)
                filho = self.mutacao(filho)
                nova_populacao.append(filho)
            populacao = np.array(nova_populacao)
            if (geracao + 1) % 10 == 0:
                print(f"Geração {geracao + 1}: Melhor Custo = R$ {melhor_fitness:.4f}")
        return self.melhor_solucao, self.melhor_custo
    
    # ---------- Impressão de resultados ----------
    def imprimir_resultados(self, solucao):
        print("\n" + "="*70)
        print("RESULTADOS DA OTIMIZAÇÃO")
        print("="*70)
        print("\nComposição Ótima do Blend:")
        for i, prop in enumerate(solucao * 100, 1):
            custo_comp = prop/100 * self.componentes.custos[i-1]
            print(f"  C{i}: {prop:6.2f}% (Custo: R$ {custo_comp:.4f}/L)")
        custo_total = self.calcular_custo(solucao)
        print(f"\nCusto Total da Mistura: R$ {custo_total:.4f}/L")
        oct, pres, benz, enx = self.calcular_propriedades(solucao)
        print("\nPropriedades da Mistura:")
        print(f"  Octanagem:        {oct:.2f} RON (Mínimo: {self.specs.octanagem_min})")
        print(f"  Pressão de Vapor: {pres:.2f} kPa (Máximo: {self.specs.pressao_vapor_max})")
        print(f"  Teor de Benzeno:  {benz:.2f}% v/v (Máximo: {self.specs.benzeno_max})")
        print(f"  Teor de Enxofre:  {enx:.2f} ppm (Máximo: {self.specs.enxofre_max})")
        print("\nVerificação de Restrições:")
        atende = True
        if oct >= self.specs.octanagem_min:
            print("  ✓ Octanagem: ATENDE")
        else:
            print("  ✗ Octanagem: NÃO ATENDE")
            atende = False
        if pres <= self.specs.pressao_vapor_max:
            print("  ✓ Pressão de Vapor: ATENDE")
        else:
            print("  ✗ Pressão de Vapor: NÃO ATENDE")
            atende = False
        if benz <= self.specs.benzeno_max:
            print("  ✓ Teor de Benzeno: ATENDE")
        else:
            print("  ✗ Teor de Benzeno: NÃO ATENDE")
            atende = False
        if enx <= self.specs.enxofre_max:
            print("  ✓ Teor de Enxofre: ATENDE")
        else:
            print("  ✗ Teor de Enxofre: NÃO ATENDE")
            atende = False
        if atende:
            print("\n  ✓ TODAS AS ESPECIFICAÇÕES FORAM ATENDIDAS!")
        else:
            print("\n  ✗ Algumas especificações não foram atendidas")
        print("="*70)
    
    # ---------- Gráfico de convergência ----------
    def plotar_convergencia(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.historico_custos)+1), self.historico_custos,
                 'b-', linewidth=2, label='Melhor Custo')
        plt.xlabel('Geração', fontsize=12, fontweight='bold')
        plt.ylabel('Custo R$/L', fontsize=12, fontweight='bold')
        plt.title('Convergência do Algoritmo Genético', fontsize=14, fontweight='bold')
        plt.grid(True)
        plt.legend()
        plt.show()


# ==============================
# Execução do AG
# ==============================
if __name__ == "__main__":
    ga = AlgoritmoGenetico(
        tamanho_pop=TAMANHO_POPULACAO,
        taxa_crossover=TAXA_CROSSOVER,
        taxa_mutacao=TAXA_MUTACAO,
        geracoes=NUM_GERACOES,
        torneio_k=TORNEIO_K
    )
    
    solucao_otima, custo_otimo = ga.executar()
    ga.imprimir_resultados(solucao_otima)
    ga.plotar_convergencia()
