import numpy as np
import pandas as pd

p_dim_debug = np.array([[404.19510562, 390.63502547, 192.64763677],
[409.96552403, 406.37891875, 195.07701505],
[402.96087877, 408.05437712, 198.28522654],
[390.47936999, 391.80649593, 205.74629947],
[396.37565655, 396.9486871,  197.21806985],
[399.24756226, 407.7053152,  206.17806707],
[403.71688256, 408.40652323, 193.51193439]])


p_dim_unknown = np.array([[394.59042841, 399.97195032, 192.82816838],
[404.070015,   398.22917185, 203.90797694],
[397.65870329, 408.181973,   202.79037281],
[402.18877541, 403.11318114, 197.89572744]])


if __name__ == '__main__':

# reading ground-truth from debug cases
    opt_p_dim = []
    for case in ['a','b','c','d','e','f','g']:
        status = 'debug'
        output = pd.read_csv('./data/pa1-'+status+'-'+case+'-output1.txt')
        heads = [int(c) for c in list(output.columns)[:-1]]
        heads = {'N_C': heads[0],'N_Frame':heads[1]}
        # print(heads)
        data = [v for v in output.values]
        opt_p_dim.append(data[1][None,:])

# ======================================= Evaluation =======================================
    error = p_dim_debug - np.concatenate(opt_p_dim, axis=0)
    errors_cases = np.linalg.norm(error,axis=1)
    # print('dimples positions of cases: ',opt_p_dim )
    print('Error of cases: ', errors_cases)
    print('Average Error: ',  np.mean(errors_cases))