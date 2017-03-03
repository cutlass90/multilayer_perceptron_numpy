import numpy as np

def sigmoid(arr):
    return 1/(1+np.exp(-arr))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def MSE(logits, targets):
    # logits shape is [batch_size, h]
    # targets shape is [batch_size, h]
    return np.square(logits - targets)

def MSE_der(logits, targets):
    # logits shape is [batch_size, h]
    # targets shape is [batch_size, h]
    return 2*(logits-targets)





class Perceptron(object):

    def __init__(self, n_inp, list_of_layers, cost_func, cost_der,
        act_func, act_der):

        self.n_inp = n_inp
        self.list_of_layers = list_of_layers
        self.cost_func = cost_func
        self.cost_der = cost_der
        self.act_func = act_func
        self.act_der = act_der

        self.n_layers = len(list_of_layers)
        l = [n_inp]+list_of_layers
        self.list_of_weights = [np.random.standard_normal(
            l[i:i+2])/100 for i in range(self.n_layers)]
        self.list_of_biases = [np.random.standard_normal([l])/100
            for l in list_of_layers]
            
    #---------------------------------------------------------------------------
    def forward_pass(self, inputs):
        # inputs shape is [batch_size, self.n_inp]
        list_of_a = []
        list_of_z = []
        for i,w in enumerate(self.list_of_weights):
            z = inputs @ w + self.list_of_biases[i]
            list_of_z.append(z)
            inputs = self.act_func(z)
            list_of_a.append(inputs)
        list_of_a[-1] = list_of_z[-1] # last layer has no activation
        return list_of_z, list_of_a

    #---------------------------------------------------------------------------
    def backward_pass(self, targets, list_of_z, list_of_a, inputs):
        # targets shape is [batch_size, list_of_layers[-1]]
        # list_of_z list len self.n_layers of arrays of shape [batch_size, h]
        # list_of_a list len self.n_layers of arrays of shape [batch_size, h]

        # error in the last layer:
        sigma_last = self.cost_der(logits=list_of_a[-1],
            targets=targets) #[batch_size, list_of_layers[-1]]
        
        list_of_sigma = [sigma_last]
        # accumulate sigma in the other layers
        for i in range(self.n_layers-1):
            sigma = list_of_sigma[-1] @ self.list_of_weights[-i-1].T *\
                self.act_der(list_of_z[-i-2]) # [b x h]
            list_of_sigma.append(sigma)
        list_of_sigma = list(reversed(list_of_sigma))

        #calculate gradients for each layer
        list_of_grads = []
        act = [inputs] + list_of_a[:-1]
        for i in range(self.n_layers):
            batch_size = targets.shape[0]
            grads = sum([list_of_sigma[i][b:b+1,...].T @ act[i][b:b+1,...]\
                for b in range(batch_size)]) / batch_size
            list_of_grads.append(grads.T)
        list_of_sigma = [s.sum(0)/batch_size for s in list_of_sigma]
        return list_of_grads, list_of_sigma

    #---------------------------------------------------------------------------
    def make_train_step(self, inputs, targets, learn_rate):
        list_of_z, list_of_a = self.forward_pass(inputs)
        list_of_grads, list_of_sigma = self.backward_pass(targets, list_of_z,
            list_of_a, inputs)

        #applying gradients
        for i in range(self.n_layers):
            self.list_of_weights[i] -= learn_rate*list_of_grads[i]
            self.list_of_biases[i] -= learn_rate*list_of_sigma[i]

    #---------------------------------------------------------------------------
    def get_cost(self, inputs, targets):
        list_of_z, list_of_a = self.forward_pass(inputs)
        cost = self.cost_func(list_of_a[-1], targets)
        return cost



if __name__ == '__main__':

    # usage
    batch_size = 1000
    n_iter = 400
    perceptron = Perceptron(n_inp = 1, list_of_layers=[10, 1], cost_func=MSE,
        cost_der=MSE_der, act_func=sigmoid, act_der=sigmoid_deriv)
    
    for it in range(n_iter):
        X = np.random.uniform(0.1,0.9,[batch_size,1])
        Y = X**3 - 2*X**2 + 10*X - 4
        print('cost', perceptron.get_cost(X, Y).mean())
        perceptron.make_train_step(X, Y, 0.1)
    



