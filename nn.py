from miniflow import *

x, y, z = Input(), Input(), Input()

#f = Add(x, y, z)

f = CaculatNode(sum, x, y, z)

feed_dict = {x: 10, y:5, z:3}
sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)


print('{} = {}'.format(' + '.join(map(lambda n: str(n.value), feed_dict)), f.value))


def mutiply(*args):
    return reduce(lambda x, y: x * y, args)


Nodes = Input(), Input(), Input(), Input(), Input()
L = [1, 2, 3, 4, 5]

feed_dict = dict()

for n, value in zip(Nodes, L):
    feed_dict[n] = value

f2 = CaculatNode(mutiply, *Nodes)
sorted_nodes = topological_sort(feed_dict)
mutiply_output = forward_pass(f2, sorted_nodes)
print('{} = {}'.format(' * '.join(map(lambda n: str(n.value), feed_dict)), f2.value))



x, y, z = Input(), Input(), Input()
weights = [0.5, 0.25, 1.4]
bias = 2

f = LinearNode([x, y, z], weights, bias)

feed_dict = {x: 6, y: 14, z: 3}
graph = topological_sort(feed_dict)
linear_output = forward_pass(f, graph)
print(linear_output)


inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(output) # should be 12.7 with this example


## Test for maxtics manipulation.

import numpy as np

X, W, b = Input(), Input(), Input()
# build the nodes

f = Linear(X, W, b)
# define the output node as a liear node.

X_ = np.array([
    [-1., -2.], 
    [-1, -2]])
W_ = np.array([
    [2., -3.], 
    [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W:W_, b:b_}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
# the output represent the output's calue, means
# f's value based on this garph.

print(output)

target = np.array([
    [-9., 4.],
    [-9., 4.]
    ])
assert(np.array_equal(output,target))


## Tset Sigmoid Node

X, W, b = Input(), Input(), Input() 

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([
    [-1., -2.],
    [-1., -2.]
    ])

W_ = np.array([
    [2., -3],
    [2., -3]
    ])

b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

target = [[  1.23394576e-04, 9.82013790e-01],
         [  1.23394576e-04, 9.82013790e-01]]

target = np.array(target)

print(output)

## MSE

y, a = Input(), Input()

cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)

forward_pass(cost, graph)


## test
np.testing.assert_array_almost_equal(output, target, decimal=8)

np.testing.assert_almost_equal(cost.value, 23.4166666667)







