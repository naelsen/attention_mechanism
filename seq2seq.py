import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_grad(y):
    return 1 - y * y

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()
  
class Softmax(object):
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.uniform(*(-1, 1), (output_size, input_size))
        self.dW = np.zeros_like(self.W)
        
        self.params = [
            ('W', self.W, self.dW),
        ]

    def initSequence(self):
        self.t = 0
        self.pred = {}
        self.x = {}
        self.targets = {}
        self.dW[:] = 0

    def forward(self, x_t):
        self.t += 1
        t = self.t

        y = self.W.dot(x_t)
        y = np.exp(y - y.max())  # for numerical stability
        y /= y.sum()

        self.pred[t] = y
        self.x[t] = x_t

        return y

    def backward(self, target):
        t = self.t

        self.targets[t] = target

        x = self.x[t]
        d = self.pred[t].copy()
        d[target] -= 1

        self.dW += np.outer(d, x)
        dh = np.dot(self.W.T, d)

        self.t -= 1

        return dh

    def getCost(self):
        return sum(-np.log(y[target]) for target, y in zip(list(self.targets.values()), reversed(list(self.pred.values())))) / len(list(self.targets.values()))

 
    class Lstm(object):
    
    def __init__(self, input_size, hidden_size, previous=None):
        if previous:
            self.previous = previous
            previous.next = self
            
        self.input_size, self.hidden_size = input_size, hidden_size

        range_unif, dim = (-1, 1), (hidden_size, hidden_size + input_size)
        # initialize weights
        self.W_f, self.b_f = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))
        self.W_i, self.b_i = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))
        self.W_ct, self.b_ct = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))
        self.W_o, self.b_o = np.random.uniform(*range_unif, dim), np.zeros((hidden_size, 1))

        # initalize gradients
        self.dW_f, self.db_f = np.zeros_like(self.W_f), np.zeros_like(self.b_f)
        self.dW_i, self.db_i = np.zeros_like(self.W_i), np.zeros_like(self.b_i)
        self.dW_ct, self.db_ct = np.zeros_like(self.W_ct), np.zeros_like(self.b_ct)
        self.dW_o, self.db_o = np.zeros_like(self.W_o), np.zeros_like(self.b_o)

        # list of all parameters
        self.params = [
            ('W_f', self.W_f, self.dW_f),
            ('W_i', self.W_i, self.dW_i),
            ('W_ct', self.W_ct, self.dW_ct),
            ('W_o', self.W_o, self.dW_o),

            ('b_f', self.b_f, self.db_f),
            ('b_i', self.b_i, self.db_i),
            ('b_ct', self.b_ct, self.db_ct),
            ('b_o', self.b_o, self.db_o)
        ]
        
        if previous:
            self.previous = previous
            previous.next = self
            
        self.initSequence()

    def initSequence(self):
        self.t = 0
        self.x = {}
        self.h = {}
        self.c = {}
        self.ct = {}

        self.forget_gate = {}
        self.input_gate = {}
        self.cell_update = {}
        self.output_gate = {}

        if hasattr(self, 'previous'):
            self.h[0] = self.previous.h[self.previous.t]
            self.c[0] = self.previous.c[self.previous.t]
        else:
            self.h[0] = np.zeros((self.hidden_size, 1))
            self.c[0] = np.zeros((self.hidden_size, 1))

        if hasattr(self, 'next'):
            self.dh_prev = self.next.dh_prev
            self.dc_prev = self.next.dc_prev
        else:
            self.dh_prev = np.zeros((self.hidden_size, 1))
            self.dc_prev = np.zeros((self.hidden_size, 1))

        # reset all gradients to zero
        for name, param, grad in self.params:
            grad[:] = 0

    def forward(self, x_t):
        self.t += 1
        x_t = x_t.reshape(-1, 1)

        t = self.t
        h = self.h[t-1]
        z = np.vstack((h, x_t))
        
        self.forget_gate[t] = sigmoid(np.dot(self.W_f, z) + self.b_f)
        self.input_gate[t] = sigmoid(np.dot(self.W_i, z) + self.b_i)
        self.cell_update[t] = tanh(np.dot(self.W_ct, z) + self.b_ct)
        self.output_gate[t] = sigmoid(np.dot(self.W_o, z) + self.b_o)

        self.c[t] = self.input_gate[t] * self.cell_update[t] + self.forget_gate[t] * self.c[t-1]
        self.ct[t] = tanh(self.c[t])
        self.h[t] = self.output_gate[t] * self.ct[t]

        self.x[t] = x_t

        return self.h[t]

    def backward(self, dh):
        t = self.t
        dh = dh.reshape(-1,1)
        
        dh = dh + self.dh_prev
        dC = tanh_grad(self.ct[t]) * self.output_gate[t] * dh + self.dc_prev

        d_forget = sigmoid_grad(self.forget_gate[t]) * self.c[t-1] * dC
        d_input = sigmoid_grad(self.input_gate[t]) * self.cell_update[t] * dC
        d_update = tanh_grad(self.cell_update[t]) * self.input_gate[t] * dC
        d_output = sigmoid_grad(self.output_gate[t]) * self.ct[t] * dh

        self.dc_prev = self.forget_gate[t] * dC

        self.db_f += d_forget
        self.db_i += d_input
        self.db_ct += d_update
        self.db_o += d_output

        z = np.vstack((self.h[t-1],self.x[t]))

        self.dW_i += np.dot(d_input, z.T)
        self.dW_f += np.dot(d_forget, z.T)
        self.dW_o += np.dot(d_output, z.T)
        self.dW_ct += np.dot(d_update, z.T)

        dz = np.dot(self.W_f.T, d_forget)
        dz += np.dot(self.W_i.T, d_input)
        dz += np.dot(self.W_ct.T, d_update)
        dz += np.dot(self.W_o.T, d_output)

        self.dh_prev = dz[:self.hidden_size]
        dX = dz[self.hidden_size:]

        self.t -= 1

        return dX
      
      
 class Embedding(object):
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = np.random.uniform(*(-1, 1), (vocab_size, embed_size))
        self.dW = np.zeros_like(self.W)

        self.params = [
            ('W', self.W, self.dW)
        ]

    def initSequence(self):
        self.t = 0
        self.x = {}
        self.dW[:] = 0

    def forward(self, x_t):
        self.t += 1
        self.x[self.t] = x_t

        return self.W[x_t]

    def backward(self, dX):
        x = self.x[self.t]
        self.dW[x] += dX.ravel()
        self.t -= 1
        
        
class Seq2seq(object):

    def __init__(self, input_size, output_size, hidden_size = None, embed_size = None, \
                 lr=0.001, clip_grad=1, curriculum_learning = (1., 0.)):
        
        if not embed_size:
            embed_size = int(np.ceil(np.max((input_size, output_size)) ** (3/4)))
            
        if not hidden_size:
            hidden_size = int(4/3 * embed_size)
        
        encoder = [
            Embedding(input_size, embed_size),
            Lstm(embed_size, hidden_size),
        ]

        decoder = [
            Embedding(output_size, embed_size),
            Lstm(embed_size, hidden_size, previous=encoder[1]),
            Softmax(hidden_size, output_size)
        ]

        self.encoder, self.decoder = encoder, decoder
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.clip_grad = clip_grad
        self.curriculum_learning = curriculum_learning

    def predict(self, X):
        if len(X) == 0:
            max_length = 0
        else:
            max_length = 2*len(X) + 1
            
        # Init to take a new sequence for the encoder
        for layer in self.encoder:
            layer.initSequence()

        # FORWARD encoder
        for x in X:
            h = x
            for layer in self.encoder:
                h = layer.forward(h)

        # Init to take a new sequence for the decoder
        for layer in self.decoder:
            layer.initSequence()

        # Output model
        out = []
        token = 0

        while len(out) < max_length:
            # Start with last generated token
            h = token

            # FORWARD decoder
            for layer in self.decoder:
                h = layer.forward(h)

            # Select token with highest softmax activation
            token = np.argmax(h)

            # Stop if we generated EOS (0) token
            if token == 0:
                break

            # Add token to output sequence
            out.append(token)

        return ' '.join([en_key2word[tok] for tok in out])

    def train_sequence(self, X, Y, curriculum_learning=1):
        # Init to take a new sequence for the encoder
        for layer in self.encoder:
            layer.initSequence()

        # FORWARD encoder
        for x in X: # Reverse X to op
            h = x
            for layer in self.encoder:
                h = layer.forward(h)

        # Init to take a new sequence for the decoder
        for layer in self.decoder:
            layer.initSequence()

        # FORWARD decoder
        for y in [0] + Y:
            if (np.random.rand() > (1 - curriculum_learning)) or (y == 0):
                h = y
            else:
                h = np.argmax(h)

            for layer in self.decoder:
                h = layer.forward(h)

        # BACKWARD decoder
        for y in reversed(Y + [0]):
            delta = y
            for layer in reversed(self.decoder):
                delta = layer.backward(delta)
                
        # BACKWARD encoder
        for x in reversed(X):
            delta = np.zeros(self.hidden_size)
            for layer in reversed(self.encoder):
                delta = layer.backward(delta)

        # Gradient clipping + sgd
        for layer in self.encoder + self.decoder:
            grad_norm = 0.0
            for name, param, grad in layer.params:
                grad_norm += np.sqrt((grad ** 2).sum())
            grad_norm = np.sqrt(grad_norm)
        
            for name, param, grad in layer.params:
                if grad_norm > self.clip_grad:
                    grad *= (self.clip_grad / grad_norm)
                param -= self.lr * grad
            
        return self.decoder[-1].getCost()

    def train(self, X_train, Y_train, epochs = 1001, verbose = 1):
        sched_forcing = np.linspace(self.curriculum_learning[0], self.curriculum_learning[1], epochs)
        for epoch in range(epochs):
            loss = 0
            for input_sequence, output_sequence in zip(X_train, Y_train):
                loss += self.train_sequence(input_sequence, output_sequence, sched_forcing[epoch])

            if epoch % 10 == 0:
                print('Epoch: ', epoch)
                print('Loss : ', loss / len(X_train))
                if verbose == 0:
                    pass
                if verbose == 1:
                    print('Checking one random prediction...')
                    input_sequence = X[np.random.randint(len(X_train))]
                    print(' '.join([fr_key2word[tok] for tok in input_sequence]), \
                          '->', seq2seq.predict(input_sequence))
                if verbose == 2:
                    print('Checking all prediction...')
                    for input_sequence in X_train:
                        print(' '.join([fr_key2word[tok] for tok in input_sequence]), \
                              '->', seq2seq.predict(input_sequence))
                print('=======================')
