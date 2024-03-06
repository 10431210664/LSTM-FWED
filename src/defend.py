import numpy as np

def feature_weighting(train_errors, test_errors, epsilon=10e-4, train=True):
    if train == True:  # 更适用于批量访问数据的场景
        weights = 1 / (epsilon + np.median(train_errors.detach().numpy(), axis=0))
    elif train == False:  # 更适用于点式接收数据的情况，如流式应用程序
        weights = 1 / (epsilon + np.median(test_errors.detach().numpy(), axis=0))
    # print("weights:", weights)
    test_errors = np.matmul(test_errors, weights)

    return test_errors
