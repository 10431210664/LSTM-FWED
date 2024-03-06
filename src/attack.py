import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import variable as var
from tqdm import trange


class Attack:
    def __init__(self, ds, model, det):
        super().__init__()
        self.ds = ds
        self.model = model
        self.det = det
        self.loss_fn = nn.MSELoss(reduction="sum")
        self.sensor_head = []
        self.header = []
        self.actuator_head = []
        self.actuator_head_MV = []
        self.actuator_head_P = []

    def generate(self, x, label, loss, threshold):
        x_ts = torch.tensor(np.array(x)).to(var.device)

        RUNS = 1
        seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
        for seed in seeds:
            GRADIENT = "E:/attack_code/LSTM-FWED/save_results/other/%s/GRADIENT_10_" % self.ds.name + ".csv"
            # NOISE_sen = "E:/attack_code/LSTM-FWED/save_results/other/%s/x_adv_only_sensor_" % self.ds.name + ".csv"
            NOISE_all = "E:/attack_code/LSTM-FWED/save_results/other/%s/x_adv_all_" % self.ds.name + ".csv"

        list_pump_status = []
        list_mv_status = []
        list_sensor_status = []

        loc1 = list(x.columns.values)  # 获取columns的值
        if self.ds.name == 'swat':
            list_pump_status = list(x.filter(
                regex='P[0-9][0-9][0-9]').columns)
            list_mv_status = list(x.filter(
                regex='MV[0-9][0-9][0-9]').columns)
            list_sensor_status_1 = [x for x in loc1 if x not in list_pump_status]
            list_sensor_status = [x for x in list_sensor_status_1 if x not in list_mv_status]

        # print("list_pump_status:", list_pump_status)
        # print("list_mv_status:", list_mv_status)
        # print("list_sensor_status:", list_sensor_status)
        x_actuator = x.drop(columns=list_sensor_status)
        x_actuator.to_csv("E:/attack_code/LSTM-FWED/data/test/test_%s_actuator.csv" % self.ds.name,
                          index=False)
        x_sensor = x[list_sensor_status]
        x_sensor.to_csv("E:/attack_code/LSTM-FWED/data/test/test_%s_sensor.csv" % self.ds.name, index=False)
        self.header = x.columns
        self.actuator_head = x_actuator.columns
        self.sensor_head = list_sensor_status
        self.actuator_head_MV = list_mv_status
        self.actuator_head_P = list_pump_status

        perturbation = 0.01
        adv_all = x_ts
        self.GetGradient(adv_all, GRADIENT)
        adv_all = self.AddNoises(adv_all, label, GRADIENT, perturbation, loss, threshold)

        reshape_adv_all = np.reshape(np.array(adv_all.cpu().numpy()), (len(adv_all), len(self.header)))
        df_adv_all = pd.DataFrame(reshape_adv_all, columns=self.header)
        df_adv_all.to_csv(NOISE_all, index=False)

        # print("归一化之后的对抗样本:\n", df_adv_all)
        # df_adv_all.to_csv(NOISE_all, index=False)
        return df_adv_all

    def GetGradient(self, test_x, GRADIENT):
        adv = []
        adv_x = test_x.cpu().clone().detach().requires_grad_(True)
        att_df = pd.DataFrame(np.array(adv_x.cpu().detach().numpy()))
        outputs = self.det.predict(self.model, att_df)
        loss = self.loss_fn(outputs.cpu(), adv_x)
        loss.backward()
        sign_grad = adv_x.grad.sign()

        df_grd_i = pd.DataFrame(np.array(sign_grad), columns=self.header)
        adv.append(np.array(df_grd_i))
        reshape_adv = np.reshape(np.array(adv), (len(test_x), len(self.header)))
        df_adv = pd.DataFrame(reshape_adv, columns=self.header)

        # Write and read
        df_adv.to_csv(GRADIENT, index=False)

        return df_adv

    # 添加扰动    
    def AddNoises(self, test_x, label, GRADIENT, perturbation, loss, threshold):
        y = np.array(label)
        abnormal_idx = np.argwhere(y == 1)  # 获取异常数据
        normal_idx = np.argwhere(y == 0)
        x_abnormal = test_x[abnormal_idx].reshape(-1, test_x.shape[1])  # 攻击之前的异常数据
        x_abnormal_df = pd.DataFrame(np.array(x_abnormal.cpu().numpy()), columns=self.header)
        x_normal = test_x[normal_idx].reshape(-1, test_x.shape[1])  # 攻击之前的异常数据
        x_normal_df = pd.DataFrame(np.array(x_normal.cpu().numpy()), columns=self.header)
        read_grad = pd.read_csv(GRADIENT)
        data_grad = np.reshape(np.array(read_grad), (len(test_x), len(self.header)))

        grad_normal = data_grad[normal_idx].reshape(-1, data_grad.shape[1])
        df_grad_normal = pd.DataFrame(grad_normal, columns=self.header)
        grad_abnormal = data_grad[abnormal_idx].reshape(-1, data_grad.shape[1])
        df_grad_abnormal = pd.DataFrame(grad_abnormal, columns=self.header)

        if self.ds.name == "swat":
            # 修改正常数据
            df_adv_sen_normal = x_normal_df[self.sensor_head] + df_grad_normal[self.sensor_head] * perturbation
            df_adv_act_mv_normal = x_normal_df[self.actuator_head_MV] + df_grad_normal[self.actuator_head_MV] * 0.5
            df_adv_act_p_normal = x_normal_df[self.actuator_head_P] + df_grad_normal[self.actuator_head_P] * 1

            # 修改异常数据
            df_adv_sen_abnormal = x_abnormal_df[self.sensor_head] - df_grad_abnormal[self.sensor_head] * perturbation
            df_adv_act_mv_abnormal = x_abnormal_df[self.actuator_head_MV] - df_grad_abnormal[
                self.actuator_head_MV] * 0.5
            df_adv_act_p_abnormal = x_abnormal_df[self.actuator_head_P] - df_grad_abnormal[self.actuator_head_P] * 1

        # 修改传感器和执行器
        df_adv_normal = pd.concat([df_adv_sen_normal, df_adv_act_mv_normal, df_adv_act_p_normal], axis=1)
        df_adv_normal = df_adv_normal[self.header]
        df_adv_normal = np.clip(df_adv_normal, 0, 1)

        df_adv_abnormal = pd.concat([df_adv_sen_abnormal, df_adv_act_mv_abnormal, df_adv_act_p_abnormal], axis=1)
        df_adv_abnormal = df_adv_abnormal[self.header]
        df_adv_abnormal = np.clip(df_adv_abnormal, 0, 1)

        adv_normal_ts = torch.tensor(np.array(df_adv_normal)).to(var.device)
        adv_abnormal_ts = torch.tensor(np.array(df_adv_abnormal)).to(var.device)

        adv_all = test_x
        for i, j in zip(normal_idx, range(len(normal_idx))):
            adv_all[i] = adv_normal_ts[j]
        for i, j in zip(abnormal_idx, range(len(abnormal_idx))):
            adv_all[i] = adv_abnormal_ts[j]

        return adv_all
