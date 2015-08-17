from __future__ import division
import numpy as np

class AdaBoost(object):

    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = np.ones(self.N)/self.N
        self.rules = []
        self.alpha = []

    def set_rule(self, func, test=False):
        errors = np.array([t[1]!=func(t[0]) for t in self.training_set])
        e = (errors*self.weights).sum()
        if test: return e
        alpha = 0.5 * np.log((1-e)/e)
        print 'e=%.2f a=%.2f'%(e, alpha)
        w = np.zeros(self.N)
        for i in range(self.N):
            if errors[i] == 1: 
                w[i] = self.weights[i] * np.exp(alpha)
            else: 
                w[i] = self.weights[i] * np.exp(-alpha)
        self.weights = w / w.sum()
        self.rules.append(func)
        self.alpha.append(alpha)

    def evaluate(self):
        NR = len(self.rules)
        for (x,l) in self.training_set:
            hx = [self.alpha[i]*self.rules[i](x) for i in range(NR)]
            print x, np.sign(l) == np.sign(sum(hx))



def rule_one(x):
    return 2*(x[0] < 1.5)-1
def rule_two(x):
    return 2*(x[0] < 4.5)-1
def rule_three(x):
    return 2*(x[1] > 5)-1

examples = []
examples.append(((1,  2  ), 1))
examples.append(((1,  4  ), 1))
examples.append(((2.5,5.5), 1))
examples.append(((3.5,6.5), 1))
examples.append(((4,  5.4), 1))
examples.append(((2,  1  ),-1))
examples.append(((2,  4  ),-1))
examples.append(((3.5,3.5),-1))
examples.append(((5,  2  ),-1))
examples.append(((5,  5.5),-1))
        
if __name__ == '__main__':

    m = AdaBoost(examples)

    m.set_rule(rule_three)
    m.set_rule(rule_two)
    m.set_rule(rule_one)

    m.evaluate()

