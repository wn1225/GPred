from sklearn.model_selection import KFold


def divide_cdhit(dataset, t_num):
    X = dataset[:t_num]
    kf = KFold(shuffle=True, random_state=1)
    # print(kf.get_n_splits(X))
    n = 0
    # fw = open('./test_new.txt', 'w')
    folds_t = []
    folds_v = []
    for train_index, vel_index in kf.split(X):
        print("t" + str(n) + '=', ','.join(str(train_index).split()) + '\n')
        print("v" + str(n) + '=',','.join(str(vel_index).split()) + '\n')
        folds_t.append(list(train_index))
        folds_v.append(list(vel_index))
        
    return folds_t,folds_v
