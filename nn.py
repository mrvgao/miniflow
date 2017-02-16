from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)

feed_dict = {x: 10, y:5, z:3}
sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)

print('{} = {}'.format(' + '.join(map(lambda n: str(n.value), feed_dict)), output))




