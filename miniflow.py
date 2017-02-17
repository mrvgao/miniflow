import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
            # set 'self' node as inbound_nodes's outbound_nodes

        self.value = None
        

    def forward(self):
        '''
        Forward propagation. 

        Compute the output value vased on 'inbound_nodes' and store the 
        result in self.value
        '''

        raise NotImplemented


class Input(Node):
    def __init__(self):
        '''
        An Input node has no inbound nodes.
        So no need to pass anything to the Node instantiator.
        '''
        Node.__init__(self)


    def forward(self, value=None):
        '''
        Only input node is the node where the value may be passed
        as an argument to forward().


        All other node implementations should get the value of the 
        previous node from self.inbound_nodes
        
        Example: 

        val0: self.inbound_nodes[0].value
        '''
        if value is not None:
            self.value = value
            ## It's is input node, when need to forward, this node initiate self's value.


        # Input subclass just holds a value, such as a data feature or a model parameter(weight/bias)


class CaculatNode(Node):
    def __init__(self, f, *nodes):
        Node.__init__(self, nodes)
        self.func = f

    def forward(self):
        self.value = self.func(map(lambda n: n.value, self.inbound_nodes))


class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)


    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inbound_nodes))
        ## when execute forward, this node caculate value as defined.



class LinearNode(Node):
    def __init__(self, nodes, weights, bias=0):
        Node.__init__(self, nodes)
        self.weights = weights
        self.bias = bias
        

    def forward(self):
        self.value = sum([n.value * w for n, w in zip(self.inbound_nodes, self.weights)])  + self.bias



class Linear(Node):
    def __init__(self, nodes, weights, bias):
        Node.__init__(self, [nodes, weights, bias])

    def forward(self):
        inbound_nodes = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value

        self.value = np.dot(inbound_nodes, weights) + bias
        #self.value = sum([n * w for n, w in zip(self.inbound_nodes[0].value, self.inbound_nodes[1].value)]) + self.inbound_nodes[2].value


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])


    def _sigmoid(self, x):
        return 1./(1 + np.exp(-1 * x))

    def forward(self):
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])


    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        assert(y.shape == a.shape)

        m = y.shape[0]

        self.value = sum([(y_i[0] - a_i[0])**2 for y_i, a_i in zip(y, a)])/m


def forward_pass(output_node, sorted_nodes):
    # execute all the forward method of sorted_nodes.

    ## In practice, it's common to feed in mutiple data example in each forward pass rather than just 1. Because the examples can be processed in parallel. The number of examples is called batch size.
    for n in sorted_nodes:
        n.forward()
        ## each node execute forward, get self.value based on the topological sort result.

    return output_node.value


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
            ## if n is Input Node, set n'value as 
            ## feed_dict[n]
            ## else, n's value is caculate as its
            ## inbounds

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

