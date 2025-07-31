# Forward pass
'''
calculate local and global gradient
select data structure - multiple dictionary
'''

## Import libraries
import numpy as np

## Defining functions
def multiply_gate(x,num):
    return x*num

def add_gate(x,y):
    return x+y

def max_gate(x,y):
    return max(x,y)


# Inputs values (x,w) and weights (y,z)
x,y,z,w = 3,-4,2,-1

#Create a dictionary of functions and required args

fwd_res = {
    'f1': 1,
    'f2': 1,
    'f3': 1,
    'f4': 1,
}

func_args = {
    'f1': (x, y),
    'f2': (z, w),
    'f3': (fwd_res['f1'], fwd_res['f2']),
    'f4': (fwd_res['f3'], 2)
}

func_dict = {
    'f1':(multiply_gate ),
    'f2': (max_gate),
    'f3': (add_gate),
    'f4': (multiply_gate) }


print('== Forward pass ==' )
def fwd_pass(func_dict):
    args_dict= {}
    for func, gate_fn in func_dict.items():
        if func == 'f1':
            args = (x, y)
        elif func == 'f2':
            args = (z, w)
        elif func == 'f3':
            args = (fwd_res['f1'], fwd_res['f2'])
        elif func == 'f4':
            args = (fwd_res['f3'], 2)

        args_dict[func] = args
        # Compute and store result
        fwd_res[func] = gate_fn(*args)
    return fwd_res, args_dict


fwdpass_res, func_args_dict = fwd_pass(func_dict)
print(f"Dict of computational units: \n{fwdpass_res}\n")
fwd_mat = np.array(list(fwdpass_res.values()))
print(f'Fwd pass matrix : {fwd_mat}')