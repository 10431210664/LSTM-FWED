import numpy as np
import pandas as pd

from LSTM_Autoencoder import LSTM_enc_dec


from evaluator import Evaluator
from real_dataset import MyDataset

RUNS = 1

def detectors(dataset_name, seed):
    standard_epochs = 20
    dets = [LSTM_enc_dec(dataset_name,num_epochs=standard_epochs, sequence_length=5, seed=seed)]

    return dets


def main():
    seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
    datasets = []
    results = pd.DataFrame()
    data_name = 'swat'
    data_train_path = 'E:/attack_code/LSTM-FWED/data/train/train_swat.csv'
    data_test_path = 'E:/attack_code/LSTM-FWED/data/test/test_swat.csv'
    dataset = MyDataset(data_name, data_train_path,
                        data_test_path)
    datasets.append(dataset)
    print('len of dbs is ' + str(len(datasets)))

    for seed in seeds:
        eva = Evaluator(datasets, detectors, seed=seed)
        result = eva.evaluate(train_model=False)  # 调用evaluator文件中的evaluateMyDB方法，评估我们的数据集

        columns = ['dataset','algorithm','accuracy','precision','recall','F1-score','auroc']
        result.to_csv("E:/attack_code/LSTM-FWED/save_results/csv_File/" + data_name +"/" +
                       str(seed) + data_name + ".csv", encoding='utf-8', columns=columns)  # 保存正常测试结果



if __name__ == '__main__':
    main()

