# Implement maxpool operation from scratch.

## Libraries
import numpy as np

## Function for CNN
def CNN_maxpool(df,ker,p,s):
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
    for x in range(0,len(mat),s):
        for y in range(0,len(mat),s):
            df_temp_1 = df_temp[x:kn+x, y:km+y]
            res = df_temp_1 * ker
            mat[x, y] = res.flatten().max()
    return mat

def main():
    ## Create dataset
    dataset = [
        [1,3,2,1,3],
        [2,9,1,1,5],
        [1,3,2,3,2],
        [8,3,5,1,0],
        [5,6,1,2,9]
    ]

    kernel = [
    [1,1,1],
    [1,1,1],
    [1,1,1]
    ]

    ## Data pre-processing
    df = np.array(dataset)
    ker = np.array(kernel)

    p = 0
    s = 1
    params = {'pad':p,'stride':s, 'df len':len(df), 'kernel_len':len(ker)}

    print('\nInput Dataset')
    print(df)

    print('\nkernel used')
    print(ker)

    print(f'\nParam used:{params}')
    print('\nNew Matrix after CNN')
    print(CNN_maxpool(df,ker,p,s))

if __name__=="__main__":
    main()