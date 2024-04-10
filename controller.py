"""A module with NAS controller-related code."""
import torch
import torch.nn.functional as F
import numpy as np
import tools
import scipy
from utils import Logger, mkdir_p
import os
import torch.nn as nn
from computational_tree import BinaryTree
import function as func
import argparse
import random
import math
import cProfile
import pstats
import scipy as sp
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import csv
# import torchdiffeq as ode
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.special import jn,jv
from scipy.spatial import KDTree
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
import sys
import functools
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
print = functools.partial(print,flush=True)

from scipy.integrate import odeint,quad

profile = cProfile.Profile()

parser = argparse.ArgumentParser('Heat Diffusion Dynamic Case')


parser.add_argument('--sampled_time',type=str,choices=['irregular','equal'],default='irregular')
parser.add_argument('--viz',action='store_true')
parser.add_argument('--adjoint',action='store_true')
parser.add_argument('--n',type=int,default=400,help='Number of nodes')
parser.add_argument('--sparse',action='store_true')
parser.add_argument('--network',type=str,choices=['grid','random','power_law','small_world','community'],default='grid')
parser.add_argument('--layout',type=str,choices=['community','degree'],default='community')
parser.add_argument('--seed',type=int,default=0,help='Random Seed')
parser.add_argument('--stop',type=float,default=100.,help='Terminal Time')
parser.add_argument('--stop_test',type=float,default=200.,help='Terminal Time test')
parser.add_argument('--operator',type=str,choices=['lap','norm_lap','kipf','norm_adj'],default='norm_lap')
parser.add_argument('--dump',action='store_true',help='Save Results')

parser.add_argument('--epoch', default=2000, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--greedy', default=0, type=float)
parser.add_argument('--random_step', default=0, type=float)
parser.add_argument('--ckpt', default='', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dim', default=3, type=int)
parser.add_argument('--tree', default='depth2', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--percentile', default=0.5, type=float)
parser.add_argument('--dsr', default=0.5, type=float)
parser.add_argument('--base', default=100, type=int)
parser.add_argument('--domainbs', default=1000, type=int)
parser.add_argument('--bdbs', default=1000, type=int)
parser.add_argument('--ft', action='store_false')
# argument for committor function

parser.add_argument('--Nepoch', type=int ,default = 5000)
# parser.add_argument('--bs', type=int ,default = 3000)
# parser.add_argument('--lr', type=float, default = 0.002)
parser.add_argument('--finetune',type=int,default=20000)

parser.add_argument('--lr_schedule', default='cos', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

unary = func.unary_functions
binary = func.binary_functions
unary_functions_str = func.unary_functions_str
unary_functions_str_leaf = func.unary_functions_str_leaf
binary_functions_str = func.binary_functions_str



def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        #print("Connected to a GPU")
    else:
        #print("Using the CPU")
        device = torch.device('cpu')
    return device





X = np.random.rand(10000, 2)
y = (np.exp(X[:,1]) + X[:,0] ** 2) ** 3

# X = np.random.uniform(-1, 1, (4000, 2))
# y = np.exp(np.sum(np.cos(X), axis=1) / 2) / 3
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Extracting x1, x2, and y values for plotting
# x1 = X[:, 0]
# x2 = X[:, 1]
# y_values = y.flatten()  # Flatten y to make it suitable for scatter plot

# # Plotting
# ax.scatter(x1, x2, y_values, color='b')

# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')

# plt.show()
# exit()


def sample_centroids_with_nbhd(X, y, num_centroids, radius):
    centroid_idx = np.random.choice(X.shape[0], num_centroids, replace=False)
    tree = KDTree(X)
    nbhds = [tree.query_ball_point(X[idx],radius) for idx in centroid_idx]
    nbhd_data = [(X[idx],y[idx]) for idx in nbhds]

    return centroid_idx, nbhd_data

# centroid_idx, nbhd_data = sample_centroids_with_nbhd(X, y, 1, 0.1)
# nbhd_data = [(coord,val) for coord, val in nbhd_data]
# print('nbhd_data',nbhd_data)

# X_ = nbhd_data[0][0]  # This is the 2D X data
# y_ = nbhd_data[0][1]  # This is the corresponding y data

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# # Plotting
# # Plot the larger dataset
# ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Larger Dataset')

# # Plot the subset
# ax.scatter(X_[:, 0], X_[:, 1], y_, color='r', label='Subset')

# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# ax.legend()

# plt.show()

# plt.show()
# exit()
class candidate(object):
    def __init__(self, action, expression, error):
        self.action = action
        self.expression = expression
        self.error = error

class SaveBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.candidates = []

    def num_candidates(self):
        return len(self.candidates)

    def add_new(self, candidate):
        flag = 1
        action_idx = None
        for idx, old_candidate in enumerate(self.candidates):
            if candidate.action == old_candidate.action and candidate.error < old_candidate.error:  # 如果判断出来和之前的action一样的话，就不去做
                flag = 1
                action_idx = idx
                break
            elif candidate.action == old_candidate.action:
                flag = 0

        if flag == 1:
            if action_idx is not None:
                # print(action_idx)
                self.candidates.pop(action_idx)
            self.candidates.append(candidate)
            self.candidates = sorted(self.candidates, key=lambda x: x.error)  # from small to large

        if len(self.candidates) > self.max_size:
            self.candidates.pop(-1)  # remove the last one

if args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

# L = 1?
elif args.tree == 'depth1':
    def basic_tree():

        tree = BinaryTree('', False)
        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree

# L = 2?
elif args.tree == 'depth2_rml':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', True)

        return tree

# L = 4?
elif args.tree == 'depth2_rmu':
    print('**************************rmu**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', False)
        tree.rightChild.insertLeft('', True)
        tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth2_rmu2':
    print('**************************rmu2**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

# L = 3
elif args.tree == 'depth2_sub':
    print('**************************sub**************************')
    def basic_tree():
        tree = BinaryTree('', True)

        tree.insertLeft('', False)
        tree.leftChild.insertLeft('', True)
        tree.leftChild.insertRight('', True)

        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth4_sub':
    def basic_tree():
        tree = BinaryTree('',True)
        tree.insertLeft('',True)
        tree.leftChild.insertLeft('',False)
        tree.leftChild.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.insertRight('',True)

        return tree

elif args.tree == 'depth5':
    def basic_tree():
        tree = BinaryTree('',True)

        tree.insertLeft('',False)

        tree.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.insertLeft('',False)
        tree.leftChild.leftChild.leftChild.insertLeft('',True)
        tree.leftChild.leftChild.leftChild.insertRight('',True)

        tree.leftChild.insertRight('',True)
        tree.leftChild.rightChild.insertLeft('',False)
        tree.leftChild.rightChild.leftChild.insertLeft('',True)
        tree.leftChild.rightChild.leftChild.insertRight('',True)

        return tree
    
structure = []

def inorder_structure(tree):
    global structure
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        inorder_structure(tree.rightChild)
inorder_structure(basic_tree())
print('tree structure', structure) # [True, False, True, True]

# 9:{0,1,Id,()^2,()^3,()^4,exp,sin,cos}, 3: {+,-,*}
structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))



if args.tree == 'depth1':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.insertRight('', True)

        return tree

elif args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

structure = []
leaves_index = []
leaves = 0
count = 0

def inorder_structure(tree):
    global structure, leaves, count, leaves_index
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        if tree.leftChild is None and tree.rightChild is None:
            leaves = leaves + 1
            leaves_index.append(count)
        count = count + 1
        inorder_structure(tree.rightChild)


inorder_structure(basic_tree())

print('leaves index:', leaves_index)

print('tree structure:', structure, 'leaves num:', leaves)

structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))
print('tree structure choices', structure_choice)

def reset_params(tree_params):
    for v in tree_params:
        # v.data.fill_(0.01)
        v.data.normal_(0.0, 0.1)

def inorder(tree, actions):
    global count
    if tree:
        inorder(tree.leftChild, actions)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary[action]
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = binary[action]
            # print(count, action, func.binary_functions_str[action])
        count = count + 1
        inorder(tree.rightChild, actions)

def inorder_visualize(tree, actions, trainable_tree,dim):
    # actions: [tensor([12]), tensor([2]), tensor([7]), tensor([1])]
    global count, leaves_cnt
    if tree:
        leftfun = inorder_visualize(tree.leftChild, actions, trainable_tree,dim)
        action = actions[count].item() # an integer

        
        if tree.is_unary:# and not tree.key.is_leave:
            if count not in leaves_index:
                midfun = unary_functions_str[action] #e.g., '({}*({})**4+{})'
                a = trainable_tree.learnable_operator_set[count][action].a.item()
                b = trainable_tree.learnable_operator_set[count][action].b.item()
            else:
                midfun = unary_functions_str_leaf[action]
        else:
            midfun = binary_functions_str[action]
        count = count + 1
        rightfun = inorder_visualize(tree.rightChild, actions, trainable_tree,dim)

        # 若 已经在叶子结点处，得到表达式
        if leftfun is None and rightfun is None:
            w = []
            for i in range(dim):
                w.append(trainable_tree.linear[leaves_cnt].weight[0][i].item())
                # w2 = trainable_tree.linear[leaves_cnt].weight[0][1].item()
            bias = trainable_tree.linear[leaves_cnt].bias[0].item()
            leaves_cnt = leaves_cnt + 1
            ## -------------------------------------- input variable element wise  ----------------------------
            expression = ''
            for i in range(0, dim):
               # print('mid fun',midfun) #(({})**4)
                x_expression = midfun.format('x'+str(i)) # e.g., (({x2})**4)
                expression = expression + ('{:.4f}*{}'+'+').format(w[i], x_expression)
            expression = expression+'{:.4f}'.format(bias)
            expression = '('+expression+')'
            return expression

        # 当 0 或者 1 在midfun 中时，由下面例子可知midfun.format 只需2个参数，反之则需要3个参数
        # unary_functions_str examples
        # '({}*(0)+{})',
        # '({}*(1)+{})',
        # # '5',
        # '({}*{}+{})',
        # # '-{}',
        # '({}*({})**2+{})',
        # '({}*({})**3+{})',

        elif leftfun is not None and rightfun is None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), leftfun, '{:.4f}'.format(b))
        elif tree.leftChild is None and tree.rightChild is not None:
            if '(0)' in midfun or '(1)' in midfun:
                return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
            else:
                return midfun.format('{:.4f}'.format(a), rightfun, '{:.4f}'.format(b))
        else:
            return midfun.format(leftfun, rightfun)
    else:
        return None

def get_function(actions):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder(computation_tree, actions)
    count = 0 # 置零
    return computation_tree

def inorder_params(tree, actions, unary_choices):
    global count
    if tree:
        inorder_params(tree.leftChild, actions, unary_choices)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary_choices[count][action]
            # if tree.leftChild is None and tree.rightChild is None:
            #     print('inorder_params:', count, action)
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = unary_choices[count][len(unary)+action]
            # print(count, action, func.binary_functions_str[action], tree.key(torch.tensor([1]), torch.tensor([2])))
        count = count + 1
        inorder_params(tree.rightChild, actions, unary_choices)

def get_function_trainable_params(actions, unary_choices):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder_params(computation_tree, actions, unary_choices)
    count = 0 # 置零
    return computation_tree

class unary_operation(nn.Module):
    def __init__(self, operator, is_leave):
        super(unary_operation, self).__init__()
        self.unary = operator
        if not is_leave:
            self.a = nn.Parameter(torch.Tensor(1).to(get_device()))
            self.a.data.fill_(1)
            self.b = nn.Parameter(torch.Tensor(1).to(get_device()))
            self.b.data.fill_(0)
        self.is_leave = is_leave

    def forward(self, x):
        if self.is_leave:
            return self.unary(x)
        else:
            return self.a*self.unary(x)+self.b

class binary_operation(nn.Module):
    def __init__(self, operator):
        super(binary_operation, self).__init__()
        self.binary = operator

    def forward(self, x, y):
        return self.binary(x, y)

leaves_cnt = 0

def compute_by_tree(tree, linear, x):
    ''' judge whether a emtpy tree, if yes, that means the leaves and call the unary operation '''
    if tree.leftChild == None and tree.rightChild == None: # leaf node
        global leaves_cnt
        transformation = linear[leaves_cnt]
        leaves_cnt = leaves_cnt + 1
        return transformation(tree.key(x))
    elif tree.leftChild is None and tree.rightChild is not None:
        return tree.key(compute_by_tree(tree.rightChild, linear, x))
    elif tree.leftChild is not None and tree.rightChild is None:
        return tree.key(compute_by_tree(tree.leftChild, linear, x))
    else:
        return tree.key(compute_by_tree(tree.leftChild, linear, x), compute_by_tree(tree.rightChild, linear, x))

class learnable_computation_tree(nn.Module):
    def __init__(self,dim):
        super(learnable_computation_tree, self).__init__()
        self.dim = dim
        self.learnable_operator_set = {}
        for i in range(len(structure)):
            self.learnable_operator_set[i] = []
            is_leave = i in leaves_index
            for j in range(len(unary)):
                self.learnable_operator_set[i].append(unary_operation(unary[j], is_leave))
            for j in range(len(binary)):
                self.learnable_operator_set[i].append(binary_operation(binary[j]))
        self.linear = []
        for num, i in enumerate(range(leaves)):
            linear_module = torch.nn.Linear(self.dim, 1, bias=True).to(get_device()) #set only one variable
            linear_module.weight.data.normal_(0, 1/math.sqrt(self.dim))
            # linear_module.weight.data[0, num%2] = 1
            linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)

    def forward(self, x, bs_action):
        # print(len(bs_action))
        global leaves_cnt
        leaves_cnt = 0
        function = lambda y: compute_by_tree(get_function_trainable_params(bs_action, self.learnable_operator_set), self.linear, y)
        out = function(x)
        leaves_cnt = 0
        return out



    
class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self,x=None):
        torch.nn.Module.__init__(self)

        self.x = x
        self.softmax_temperature = 5.0
        self.tanh_c = 2.5
        self.mode = True

        self.input_size = 20
        self.hidden_size = 50
        self.output_size = sum(structure_choice) # e.g., 39, i.e., sum of [12, 3, 12, 12]


        if self.x is None:
            self._fc_controller = nn.Sequential(
                nn.Linear(self.input_size,self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size,self.output_size))
            
        else:
            self._fc_controller = nn.Sequential(nn.Linear(self.x.size(-1),self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.output_size))

    def forward(self,x):


        logits = self._fc_controller(x)


        logits /= self.softmax_temperature

        # exploration # ??
        if self.mode == 'train':
            logits = (self.tanh_c*F.tanh(logits))

        return logits

    def sample(self,x,batch_size=1, step=0):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        # [B, L, H]
        if x is None:
            inputs = torch.zeros(batch_size, self.input_size).to(get_device())
        else:
            # x = self.x.unsqueeze(1)
            print('x',x.size()) #torch.Size([num_x, dim_x])
            x_batch = repeat(x, 'n d -> b n d', b = batch_size)
            inputs = rearrange(x_batch, 'b n d -> (b n) d').to(get_device())
            print('x_batch',x_batch.size()) # [batch_size, num_x, dim_x]
            print('inputs',inputs.size()) # [batch_size * num_x, dim_x], e.g., [200,2]





        log_probs = []
        actions = []
        total_logits = self.forward(inputs) # (10*20,36)
        total_logits = rearrange(total_logits,'(b n) d -> b n d', b = batch_size) # (10,20,36)


        cumsum = np.cumsum([0]+structure_choice) # [ 0 11 14 25 36]

        for i in range(x.size(0)): # for each centroid
            for idx in range(len(structure_choice)): # structure_choice : [11 3 11 11], so idx 0,1,2,3
                logits = total_logits[:,i, cumsum[idx]:cumsum[idx+1]] # (10, 11)

                probs = F.softmax(logits, dim=-1) # tensor of size 10 * 11
                action = probs.multinomial(num_samples=1).data # (10,1)




                # log prob of sampling a particular operator for a particular node of tree
                log_prob = F.log_softmax(logits, dim=-1) # tensor of size  10 * 11
                # print(probs)

                if step >= args.random_step:
                    action = probs.multinomial(num_samples=1).data # (batch_size,1)

                else:
                    action = torch.randint(0, structure_choice[idx], size=(batch_size, 1)).to(get_device())

                # action is a tensor of size (batch_size,1),i.e., 对batch 中每个树的 第idx 节点，选择一个operator
                # 对batch 中每一个树，按照概率 sample 其中一个节点的action，最后action 大小为（batch_size,1)

                # 以epsilon的概率均匀sample,i.e., batch 中所有树的第idx 节点都选择同样的均匀sample出的节点
                if args.greedy != 0:
                    for k in range(args.bs):
                        if np.random.rand(1) < args.greedy:
                            choice = random.choices(range(structure_choice[idx]), k=1)
                            action[k] = choice[0]
                # selected_log_prob of size (batch_size, 1)
                selected_log_prob = log_prob.gather(
                    1, tools.get_variable(action, requires_grad=False)) # (batch_size,1)

                
                log_probs.append(selected_log_prob[:, 0:1])
                actions.append(action[:, 0:1])



        # (batch_size,number of nodes * number of centroids),e.g., (10,20*4)
        log_probs = torch.cat(log_probs, dim=-1)   
        actions = torch.cat(actions,dim=-1)

        log_probs = rearrange(log_probs,'b (x y) -> b x y',x = x.size(0),y = len(structure_choice))
        actions = rearrange(actions,'b (x y) -> b x y',x = x.size(0),y = len(structure_choice))


        return actions, log_probs # both of size (batch_size, number of centroids, number of nodes)

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))



def get_reward(centroids,nbhd_data,bs, actions, learnable_tree,\
             tree_params):



    centroids.requires_grad = True
    criterion = nn.MSELoss()
    # print(x)
    regression_errors = []
    formulas = []


    batch_size = bs  # 10

    global count, leaves_cnt

    # 做T1+T2的意义是，即使给定了一组🌲的operator set，仍需要学习每个节点的weight and bias,i.e., trainable parameters

    for i in range(centroids.size(0)):
        for bs_idx in range(batch_size):
            # bs_action 是 一个batch 中完整一棵树的4个节点的分别的action
            ## keep tree1 and tree2 the same format
            # actions = actions.view(batch_size, xnew.size(0), -1) 

            bs_action = actions[bs_idx,i,:].view(-1,1) # [num_nodes,1]
            bs_action = [torch.tensor([item]) for item in bs_action] # [tensor([3]), tensor([0]), tensor([4])]


            reset_params(tree_params)
            tree_optim = torch.optim.Adam(tree_params, lr=0.005)

            
            for _ in range(T1):
                local_loss = criterion(learnable_tree(nbhd_data[i][0],bs_action),nbhd_data[i][1])
                tree_optim.zero_grad()
                local_loss.backward()
                tree_optim.step()

            tree_optim = torch.optim.LBFGS(tree_params, lr=1, max_iter=T2)
            print('----------- centroid idx {} ************* batch idx {} -----------'.format(i,bs_idx))


            error_hist = [] #error_hist when doing LBFGS

            def closure():
                tree_optim.zero_grad()
                local_loss = criterion(learnable_tree(nbhd_data[i][0],bs_action),nbhd_data[i][1])

                # print('loss before: ', loss.item())
                error_hist.append(local_loss.item())
                local_loss.backward()
                return local_loss

            tree_optim.step(closure)

            local_loss = criterion(learnable_tree(nbhd_data[i][0],bs_action),nbhd_data[i][1])

            regression_error = local_loss


            error_hist.append(regression_error.item())

            print(' min: ', min(error_hist))
            regression_errors.append(min(error_hist))

            count = 0
            leaves_cnt = 0
            formula = inorder_visualize(basic_tree(), bs_action, learnable_tree,dim=2)
            count = 0
            leaves_cnt = 0

            formulas.append(formula)


    return regression_errors, formulas

    

def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

def true(x):
    return -0.5*(torch.sum(x**2, dim=1, keepdim=True))

def best_error(nbhd_data,best_action,learnable_tree):

    criterion = nn.MSELoss()


    xnew,dy = nbhd_data[0][0],nbhd_data[0][1]

    dy = torch.from_numpy(dy).to(get_device()).to(torch.float32)
    
    xnew = torch.from_numpy(xnew).to(get_device()).to(torch.float32)
    xnew.requires_grad = True

    # keep tree1 and tree2 same format
    bs_action = best_action

    # local_loss = criterion(learnable_tree(nbhd_data[i][0],bs_action),nbhd_data[i][1])
    
    loss = criterion(learnable_tree(xnew,bs_action),dy)

    regression_error = loss

    return regression_error


def train_controller(xtest,dytest,controller, controller_optim,\
     learnable_tree, tree_params,\
          hyperparams):

    ### obtain a new file name ###



    file_name = os.path.join(hyperparams['checkpoint'], 'tree_{}_log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(1,file_idx)):
        file_idx += 1
    file_name = file_name.format(1,file_idx)
    logger_tree = Logger(file_name, title='')
    logger_tree.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])

    controller.train()
    baseline = None

    bs = args.bs
    smallest_error = float('inf')


    pool_size = 10

    candidates = SaveBuffer(pool_size)



    for step in range(hyperparams['controller_max_step']):
        # sample models
        centroid_idx, nbhd_data = sample_centroids_with_nbhd(xtest,dytest,num_centroids=hyperparams['num_centroid'],radius=hyperparams['radius'])

        centroids = torch.tensor(xtest[centroid_idx]).to(get_device()).to(torch.float32) # torch.Size([num_centroid, x_dim])

        nbhd_data = [(torch.tensor(coords).to(get_device()).to(torch.float32),torch.tensor(values).to(get_device()).to(torch.float32)) for coords,values in nbhd_data]
        actions, log_probs = controller.sample(centroids,batch_size=bs, step=step)# (batch_size, num_x * num_nodes), i.g., 10 by 80


        rewards,formulas = get_reward(centroids,nbhd_data,bs, actions, learnable_tree,\
             tree_params) # rewards is a list of batch_size * num_centroids errors
        
        
        rewards = torch.FloatTensor(rewards).to(get_device()).view(len(centroids),-1) # torch.Size([num_centroids,batch_size]),e.g.,20 by 10





        # discount
        if 1 > hyperparams['discount'] > 0:
            rewards = discount(rewards, hyperparams['discount'])

        base = args.base
        rewards[rewards > base] = base
        rewards[rewards != rewards] = 1e10
        error = rewards # size (bs,1)

        # 此时的rewards 为真正的rewards，i.e., 越大越好

        rewards = 1 / (1 + torch.sqrt(rewards))


 
        # moving average baseline
        if baseline is None:
            baseline = (rewards).mean()
        else:
            decay = hyperparams['ema_baseline_decay'] # 0.95
            baseline = decay * baseline + (1 - decay) * (rewards).mean()


        argsort = torch.argsort(rewards,dim=1, descending=True)  # (num_centroids, batch_size) # 20 by 10
        print('argsort',argsort)



        num = int(args.bs * args.percentile) # 5
        rewards_sort = torch.gather(rewards, 1, argsort) # (num_centroids, batch_size)





        adv = rewards_sort - rewards_sort[:,num:num + 1]  # - baseline, (num_centroids, batch_size)


        log_probs_sort = torch.empty_like(log_probs) # (10,20,4)
        for i in range(log_probs.size(1)):
            indices = argsort[i] # (batch_size,)
            # print('indices',indices)
            # print('log_probs',log_probs[:,i,:])
            log_probs_sort[:,i,:] = log_probs[:,i,:][indices]

 


        # print('adv', adv)
        print()

        ################### Loss in the form of expectation ###################### # loss is of size (num_centroids, num,nodes)
        loss = -(log_probs_sort[:num, :,:].transpose(0,1)) * tools.get_variable(adv[:,:num].unsqueeze(-1), False, requires_grad=False)
        # loss = (loss.sum(dim=1)).mean(dim=1).mean()     
        loss = loss.sum(dim=1).sum(dim=1).mean()


        # update
        controller_optim.zero_grad()
        loss.backward()
        if hyperparams['controller_grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(controller.parameters(),
                                          hyperparams['controller_grad_clip'])
        controller_optim.step()



        if (step + 1) % 50 == 0:
                # Save the model every 100 epochs

            torch.save(controller.state_dict(), f'model_epoch_{step+1}.pth')
            # Optional: also save optimizer state
            torch.save(controller_optim.state_dict(), f'optimizer_epoch_{step+1}.pth')


def fine_tune(xtest,dytest,controller,
     learnable_tree, 
          hyperparams,model_path):
    
    global count, leaves_cnt
    # model = Controller(xtest).to(get_device())
    # model.load_state_dict(torch.load(model_path))
    # model.eval()

    file_name = os.path.join(hyperparams['checkpoint'], 'tree_{}_log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(1,file_idx)):
        file_idx += 1
    file_name = file_name.format(1,file_idx)
    logger_tree = Logger(file_name, title='')
    logger_tree.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])

    controller.load_state_dict(torch.load(model_path))
    controller.eval()


    pool_size = 10

    candidates = SaveBuffer(pool_size)
    x = np.random.rand(1,2)
    # x = np.random.uniform(-1, 1, (1, 2))
    x = torch.tensor(x).to(torch.float32)
    xtest = torch.tensor(xtest,dtype=torch.float32)
    print('x',x)
    tree = KDTree(xtest)
    nbhds = tree.query_ball_point(x,0.1)
    nbhd_data = [(X[idx],y[idx]) for idx in nbhds]


    actions, _ = controller.sample(x,batch_size=20, step=3) 
    actions = rearrange(actions,'b 1 d -> b d')

    binary_code = ''
    for action in actions:
        bs_action = [torch.tensor([item]) for item in action]
        
        # print('bs_action',bs_action)
        # exit()

        # binary_code = binary_code + str(action[0].item())

        count = 0
        leaves_cnt = 0
        formula = inorder_visualize(basic_tree(), bs_action, learnable_tree,dim=2)
        count = 0
        leaves_cnt = 0
        print('formula',formula)

        b_error = best_error(nbhd_data,bs_action,learnable_tree)
        candidates.add_new(candidate(action=bs_action, expression=formula, error=b_error))

    exit()


    for candidate_ in candidates.candidates:
        print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action],
                                                    candidate_.expression))
        action_string = ''
        for v in candidate_.action:
            action_string += str(v.item()) + '-'
        logger_tree.append([666, 0, 0, action_string, candidate_.error.item(), candidate_.expression])


    


        # logger.append([666, 0, 0, 0, candidate_.error.item(), candidate_.expression]) 


    ############## Finetune candidates in the pool
    finetune = args.finetune

    # global count, leaves_cnt
    for candidate_ in candidates.candidates:
        trainable_tree = learnable_computation_tree(dim=2)
        trainable_tree = trainable_tree.to(get_device())

        params = []
        for idx, v in enumerate(trainable_tree.learnable_operator_set):
            if idx not in leaves_index:
                for modules in trainable_tree.learnable_operator_set[v]:
                    for param in modules.parameters():
                        params.append(param)
        for module in trainable_tree.linear:
            for param in module.parameters():
                params.append(param)


        reset_params(params)
        tree_optim = torch.optim.Adam(params, lr=1e-2)

        
        for current_iter in range(finetune):
            error = best_error(nbhd_data,candidate_.action,\
                                trainable_tree)

            tree_optim.zero_grad()
            error.backward()
            if hyperparams['finetune_grad_clip'] > 0:

                torch.nn.utils.clip_grad_norm_(params,
                    max_norm = hyperparams['finetune_grad_clip'])

            tree_optim.step()

            count = 0
            leaves_cnt = 0
            formula = inorder_visualize(basic_tree(),candidate_.action,trainable_tree,dim=2)
            leaves_cnt = 0
            count = 0



            suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(current_iter=current_iter, error=error, formula=formula)
            if (current_iter + 1) % 100 == 0:
                logger_tree.append([current_iter, 0, 0, 0, error.item(), formula])

            if args.lr_schedule == 'cos':
                cosine_lr(tree_optim, 1e-2, current_iter, finetune)
            elif args.lr_schedule == 'exp':
                expo_lr(tree_optim,1e-2,current_iter,gamma=0.999)



        if isinstance(xtest, np.ndarray):
            xtest = torch.from_numpy(xtest).to(get_device()).float()
        if isinstance(dytest,np.ndarray):
            dytest = torch.from_numpy(dytest).to(get_device()).float()

        # ✅

        criterion = nn.MSELoss()
        loss = criterion(learnable_tree(xtest,candidate_.action),dytest)

        

        print('l2 error: ', loss.item())

        logger_tree.append(['l2 error', 0, 0, 0, loss.item(), 0])



def expo_lr(opt,base_lr,e,gamma):
    lr = base_lr * gamma ** e
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':


    sampled_points = torch.tensor(X).to(torch.float32)


    controller = Controller(sampled_points).to(get_device())


    hyperparams = {}
    hyperparams['ft'] = False
    hyperparams['controller_max_step'] = args.epoch
    hyperparams['discount'] = 1.0
    hyperparams['ema_baseline_decay'] = 0.95
    hyperparams['controller_lr'] = args.lr
    hyperparams['entropy_mode'] = 'reward'
    hyperparams['controller_grad_clip'] = 0 #10
    hyperparams['finetune_grad_clip'] = 0
    hyperparams['checkpoint'] = args.ckpt
    hyperparams['num_centroid'] = 30
    hyperparams['radius'] = 0.3

    if not os.path.isdir(hyperparams['checkpoint']):
        mkdir_p(hyperparams['checkpoint'])


    controller_optim = torch.optim.Adam(controller.parameters(), lr= hyperparams['controller_lr'])

    trainable_tree = learnable_computation_tree(dim=2).to(get_device())



    


    params = []
    for idx, v in enumerate(trainable_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for module in trainable_tree.linear:
        for param in module.parameters():
            params.append(param)




    T1 = 20
    T2 = 20

print('ft',hyperparams['ft'])
if hyperparams['ft']:
    fine_tune(X,y,controller,trainable_tree, hyperparams,model_path='model_epoch_500.pth')
else:
    train_controller(X,y,controller, controller_optim,trainable_tree, params, hyperparams)