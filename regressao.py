print("Guilherme dos Santos")

import numpy as np

def f_true(x):
    return 2 + 0.8 * x

# conjunto de dados {(x,y)}
xs = np.linspace(-3, 3, 100)
ys = np.array( [f_true(x) + np.random.randn()*0.5 for x in xs] )


''' hipotese 
'''
def h(x, theta):
    pass


''' funcao de custo 
'''
def J(theta, xs, ys):
    pass


''' derivada parcial com respeito a theta[i] 
'''
def gradient(i, theta, xs, ys):
    pass


''' plota no mesmo grafico: - o modelo/hipotese (reta)
    - a reta original (true function)
    - e os dados com ruido (xs, ys) 
'''
def print_modelo(theta, xs, ys):
    pass


theta = # preencher
alpha = # preencher
epochs = 5000

custo = []

for i in range(epochs): # 10000

    t0 = # preencher com o novo theta[0]
    t1 = # preencher com o novo theta[1]
    theta[0] = t0
    theta[1] = t1
    custo.append( J(theta, xs, ys) )
    
    if i % 5 == 0:
        print_modelo(theta, xs, ys)
    

## -
plot(custo, '.-')
show()
