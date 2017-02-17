import random

def f(x):
    return x**2 + 5


def df(x):
    return 2*x


old_x = float('inf')
x = random.randint(0, 10000)
learning_rate = 0.3
epochs = 0


while abs(x - old_x) > 1.0e-7:
    cost = f(x)
    gradx = df(x)
    ## grad takes into account the effect each parameter has on the cost, so 
    ## that's hwo to find the direction of steepest ascent.

    old_x = x
    x -= learning_rate * gradx

    print('EPOCH{}: Cost={:.3f}, x = {:.3f}'.format(epochs, cost, gradx))
    epochs += 1

    
