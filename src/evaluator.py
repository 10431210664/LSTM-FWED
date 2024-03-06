import gc
import sys
import os
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as skm

from LSTM_Autoencoder import LSTMEDModule
from real_dataset import create_train_set

import variable as var
import defend

from attack import Attack


class Evaluator:
    def __init__(self, datasets: list, detectors: callable, output_dir: str = None, seed: int = None):
        """
        :param datasets: list of datasets
        :param detectors: callable that returns list of detectors可调用返回检测器列表,即采用的哪种检测算法
        """

        assert np.unique([x.name for x in datasets]).size == len(datasets), 'Some datasets have the same name!'
        self.datasets = datasets
        self._detectors = detectors
        self.results = dict()
        self.output_dir = output_dir or 'save_results'
        # Dirty hack: Is set by the main.py to insert results from multiple evaluator runs由 main.py 设置以插入来自多个评估器运行的结果
        self.benchmark_results = None
        # Last passed seed value in evaluate()
        self.seed = seed
        self.train_loss = []
        self.test_featurewise_losses = None

    def benchmarks(self, score) -> pd.DataFrame:
        df = pd.DataFrame()
        for ds in self.datasets:
            _, _, _, y_test = ds.data()
            for det in self.detectors:
                print("det.name:", det.name)
                threshold = self.get_optimal_threshold(y_test, np.array(score))

                print("threshold:", threshold)
                y_pred = self.binarize(score, threshold)
                # y_pred = self.predicted_lables(threshold, score)
                tp, fp, tn, fn = self.get_metrics_values(y_test, y_pred)
                print("tp={},fp={},tn={},fn={}".format(tp, fp, tn, fn))
                accuracy, precision_score, recall_score, f1_score = self.get_accuracy_precision_recall_fscore(y_test, y_pred)
                auc = self.get_auroc(ds, score)
                print("acc={}, prec={}, rec={}, f1_score={}, auc={}".format(accuracy,precision_score, recall_score, f1_score,auc))
                tmp = pd.DataFrame({'dataset': ds.name,
                                'algorithm': det.name,
                                'threshold': threshold,
                                'accuracy':accuracy,
                                'precision': precision_score,
                                'recall': recall_score,
                                'F1-score': f1_score,
                                'auroc': auc,
                                'TP': tp,
                                'FP': fp,
                                'TN': tn,
                                'FN': fn}, index=[0])#默认是垂直添加，axis=0 
                df=pd.concat([df,tmp],ignore_index=True)
        return df

    def evaluate(self, train_model):
        for ds in self.datasets:  # ds:swat/wadi
            # results = dict()
            results = pd.DataFrame()
            loss_fn = nn.SmoothL1Loss(reduction='none')
            (X_train, _, X_test, y_test) = ds.data()  # 调用real_datasets函数里的data方法，处理训练集和测试集的数据和标签
            for det in self.detectors:  # 调用检测器，det:检测器名字
                if det.name == "LSTM-ED":
                    net = LSTMEDModule(X_train.shape[1], var.hidden_size,
                                    var.n_layers, var.use_bias, var.dropout,
                                    seed=self.seed, gpu=var.gpu)
                net = det.fit(net, X_train.copy())  # 用训练集进行训练检测器

                if det.name=="LSTM-ED":
                    # 计算训练集的重构误差
                    losses_fw = []
                    train_loader, _ = create_train_set(X_train)
                    for x_batch in train_loader:
                        x_rec = net(x_batch.to(var.device))
                        loss = loss_fn(x_batch, x_rec.cpu())
                        losses = loss.mean()
                        loss_fw = loss.mean(1)
                        self.train_loss.append(losses.item())
                        losses_fw.append(loss_fw)
                    train_losses = torch.cat(losses_fw)

                    X_test_nm = ds.Normalized(X_test)

                    outputs = det.predict(net, X_test_nm)

                    test_x_ts = torch.tensor(np.array(X_test_nm))
                    test_ft_losses = loss_fn(test_x_ts, outputs)
                    test_losses = test_ft_losses.mean(1)
                    score_normal = np.array(test_losses)
                    print("------------Action origin evaluator-------------")
                    result_normal = self.benchmarks(score_normal)
                    results = pd.concat([results, result_normal], ignore_index=True)
            
                    # 攻击attack
                    adversarial = Attack(ds, net, det)
                    att_x_test = adversarial.generate(X_test_nm, y_test,test_ft_losses,result_normal.threshold)

                    outputs = det.predict(net, att_x_test)
                    test_x_ts = torch.tensor(np.array(att_x_test))
                    self.test_featurewise_losses = loss_fn(outputs, test_x_ts)
                    att_losses = self.test_featurewise_losses.mean(1)
                    att_score = np.array(att_losses)
                    print("------------Action attack evaluator-------------")
                    result_attack = self.benchmarks(att_score)
                    results = pd.concat([results, result_attack], ignore_index=True)
                    
                    # 防御
                    # feature weighting  
                    FW_defend_losses = defend.feature_weighting(train_losses, self.test_featurewise_losses, epsilon=var.epsilon,
                                                                train=True)
                    FW_defend_score = np.array(FW_defend_losses)
                    print("------------Action FW_defend evaluator-------------")
                    result_FW_defend = self.benchmarks(FW_defend_score)
                    results = pd.concat([results, result_FW_defend], ignore_index=True)
                else:
                    ''''''


        gc.collect()
        return results



    @property
    def detectors(self):
        for ds in self.datasets:
            detectors = self._detectors(ds.name,self.seed)
        assert np.unique([x.name for x in detectors]).size == len(detectors), 'Some detectors have the same name!'
        return detectors

    @staticmethod
    def get_metrics_values(y_test, y_pred):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for yIndex in range(len(y_pred)):
            if y_pred[yIndex] == y_test[yIndex]:
                if y_test[yIndex] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if y_test[yIndex] == 0:
                    fp += 1
                else:
                    fn += 1
        return tp, fp, tn, fn

    @staticmethod
    def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        return accuracy, precision, recall, f_score

    @staticmethod
    def get_auroc(ds, score):
        if np.isnan(score).all():
            score = np.zeros_like(score)
        _, _, _, y_test = ds.data()
        score_nonan = score.copy()
        score_nonan[np.isnan(score_nonan)] = np.nanmin(score_nonan) - sys.float_info.epsilon
        fpr, tpr, _ = roc_curve(y_test, score_nonan)
        return auc(fpr, tpr)
        
    def get_optimal_threshold(self, y_test, score, steps=100, return_metrics=False):  # 返回f_score值最高对应的阈值
        maximum = np.nanmax(score)
        minimum = np.nanmin(score)
        threshold = np.linspace(minimum, maximum, steps)
        metrics = list(
            self.get_metrics_by_thresholds(y_test, score, threshold))  #
        metrics = np.array(metrics).T
        anomalies, acc, prec, rec, f_score = metrics
        if return_metrics:
            return anomalies,acc, prec, rec, f_score, threshold
        else:
            return threshold[np.argmax(f_score)]

    def get_metrics_by_thresholds(self, y_test: list, score: list, thresholds: list):
        for threshold in thresholds:
            anomaly = self.binarize(score, threshold=threshold)
            metrics = Evaluator.get_accuracy_precision_recall_fscore(y_test, anomaly)
            yield (anomaly.sum(), *metrics)

    def binarize(self, score, threshold=None):
        threshold = threshold if threshold is not None else self.threshold(score)
        score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
        return np.where(score >= threshold, 1, 0)

    def threshold(self, score):
        return np.nanmean(score) + 2 * np.nanstd(score)


