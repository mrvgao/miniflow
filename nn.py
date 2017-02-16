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
