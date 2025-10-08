# Implement convolution operations from scratch. Assume a 3x3 kernel and apply it on an input image of 32x32.

## Libraries
import pandas as pd
import numpy as np

## Function for CNN
def CNN(df,ker,p,s):
    '''

    :param df: Input Dataframe
    :param ker: Filter used
    :return: generated matrix after using kernel
    '''

    kn,km = ker.shape # saving shape of kernel

    # Creating empty mat using formula
    n = len(df)
    f = len(ker)
    mat_dim = ((n + 2*p - f)/s) +1
    mat = np.zeros((int(mat_dim),int(mat_dim)))
    df_temp = np.pad(df,(p,p), mode="constant") # extending mat with pad ==p


    # Initiating matrix calculation
    for x in range(0,len(mat)):
        for y in range(0,len(mat)):
            df_temp_1 = df_temp[x*s:kn+x*s, y*s:km+y*s]
            res = df_temp_1 * ker
            mat[x, y] = res.flatten().sum()
    return mat

def main():
    ## Create dataset
    dataset = [
    [3,0,1,2,7,4],
    [1,5,8,9,3,1],
    [2,7,2,5,1,3],
    [0,1,3,1,7,8],
    [4,2,1,6,2,8],
    [2,4,5,2,3,9]
    ]

    kernel = [
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
    ]

    ## Data pre-processing
    df = np.array(dataset)
    ker = np.array(kernel)

    p = 0
    s = 2
    params = {'pad':p,'stride':s, 'df len':len(df), 'kernel_len':len(ker)}

    print('\nInput Dataset')
    print(df)

    print('\nkernel used')
    print(ker)

    print(f'\nParam used:{params}')
    print('\nNew Matrix after CNN')
    print(CNN(df,ker,p,s))

    print('\nWorking on the image')
    df = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
    print(CNN(df,ker,p,s))



if __name__=="__main__":
    main()