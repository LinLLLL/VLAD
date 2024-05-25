import os
import numpy as np
import linecache
import pandas as pd
import xlsxwriter

test_ID = False
ACC_all = {}
# for domain in ["test"]:
#     root = '/data_disk/zl/IJCV_Rebuttal_abs/CelebA_abla_nk/FewShot_ViT/' + domain + '/VLAD/'
for domain in ["ColoredCatsDogs.json"]:
    root = '/data_disk/zl/IJCV_Rebuttal_L/ColoredCatsDogs/FewShot_ViT/' + domain + '/VLAD/'
    key = domain[:-4].split('_')[-1]
    ACC = {}
    P0 = os.listdir(root)
    for p0 in P0:
        key0 = p0[-1]  # seed
        ACC[key0] = {}
        path0 = os.path.join(root, p0)
        print(path0)
        P1 = os.listdir(path0)
        for p1 in P1:
            key1 = p1 #.split('_')  # hps
            ACC[key0][key1] = []
            path1 = os.path.join(path0, p1)
            log_path = os.path.join(path1, "log.txt")

            line = linecache.getline(log_path, 400)
            i = 1
            while 'Deploy the model with the best val performance' not in line:
                line = linecache.getline(log_path, 400 + i)
                i += 1
                if i > 5000:
                    break

            for j in range(6):
                get_epoch = linecache.getline(log_path, 400 + i + j)
                if 'epoch' in get_epoch:
                    epoch = int(get_epoch.split('(')[-1].split(')')[0].split('=')[-1].strip(' '))
                    break
            if test_ID:
                for k in range(16):
                    line = linecache.getline(log_path, 400 + i + k)
                    if 'accuracy' in line:
                        ACC[key0][key1] = [float(line.split(':')[1].strip().split('%')[0])]
                        line = linecache.getline(log_path, 400 + i + k + 8)
                        ACC[key0][key1].append(float(line.split(':')[1].strip().split('%')[0]))
                        line = linecache.getline(log_path, 400 + i + k + 9)
                        if "None" not in line:
                            ACC[key0][key1][key2].append(float(line.split(':')[1].strip().split('%')[0]))
                        else:
                            ACC[key0][key1][key2].append(0)
                        line = linecache.getline(log_path, 400 + i + k + 10)
                        ACC[key0][key1].append(float(line.split(':')[1].strip().split('%')[0]))
                        line = linecache.getline(log_path, 400 + i + k + 3)
                        ACC[key0][key1].append(float(line.strip('\n').split(':')[1].strip()))
                        line = linecache.getline(log_path, 400 + i + k + 4)
                        ACC[key0][key1].append(float(line.strip('\n').split(':')[1].strip()))
                        line = linecache.getline(log_path, 400 + i + k + 5)
                        ACC[key0][key1].append(float(line.strip('\n').split(':')[1].strip().split(' ')[1]))
                        line = linecache.getline(log_path, 400 + i + k + 6)
                        ACC[key0][key1].append(float(line.strip('\n').split(' ')[-1]))
                        line = linecache.getline(log_path, 400 + i + k + 7)
                        ACC[key0][key1].append(float(line.strip('\n').split(' ')[-1]))
                        break
            else:
                with open(log_path) as f:
                    idx = 1
                    t = 0
                    for line in f.readlines():
                        if 'Do evaluation on test set' in line:
                            t += 1
                            if t == 30:
                                acc_line = linecache.getline(log_path, idx + 4)
                                ACC[key0][key1] = [float(acc_line.split(':')[-1].split('%')[0].strip())]
                                line = linecache.getline(log_path, idx + 4 + 9)
                                ACC[key0][key1].append(float(line.split(':')[1].strip().split('%')[0]))
                                line = linecache.getline(log_path, idx + 4 + 10)
                                if "None" not in line:
                                    ACC[key0][key1].append(float(line.split(':')[1].strip().split('%')[0]))
                                else:
                                    ACC[key0][key1].append(0)
                                line = linecache.getline(log_path, idx + 4 + 11)
                                if "None" not in line:
                                    ACC[key0][key1].append(float(line.split(':')[1].strip().split('%')[0]))
                                else:
                                    ACC[key0][key1].append(0)
                                line = linecache.getline(log_path, idx + 4 + 3)
                                ACC[key0][key1].append(float(line.strip('\n').split(':')[1].strip()))
                                line = linecache.getline(log_path, idx + 4 + 4)
                                ACC[key0][key1].append(float(line.strip('\n').split(':')[1].strip()))
                                line = linecache.getline(log_path, idx + 4 + 5)  #########################
                                ACC[key0][key1].append(float(line.strip('\n').strip().split(' ')[1]))  ##########
                                line = linecache.getline(log_path, idx + 4 + 6)
                                ACC[key0][key1].append(float(line.strip('\n').split(':')[1].strip().split(' ')[1]))
                                line = linecache.getline(log_path, idx + 4 + 7)
                                ACC[key0][key1].append(float(line.strip('\n').split(' ')[-1]))
                                line = linecache.getline(log_path, idx + 4 + 8)
                                ACC[key0][key1].append(float(line.strip('\n').split(' ')[-1]))
                                break
                        idx += 1
            # get best model's val_acc (few-shots of val set)
            with open(log_path) as f:
                idx = 1
                for l in f.readlines():
                    if ('epoch [' + str(30)) in l:
                        for k in range(30):
                            acc_line = linecache.getline(log_path, idx + k)
                            if 'Do evaluation on val set' in acc_line:
                                acc_line = linecache.getline(log_path, idx + k + 4)
                                ACC[key0][key1].append(float(acc_line.split(':')[-1].split('%')[0].strip()))
                                break
                        break
                    idx += 1
            # get model's test_acc at the last training step
            with open(log_path) as f:
                idx = 1
                for l in f.readlines():
                    if ('epoch [' + str(30)) in l:
                        for k in range(20):
                            acc_line = linecache.getline(log_path, idx + k)
                            if 'Do evaluation on test set' in acc_line:
                                acc_line = linecache.getline(log_path, idx + k + 4)
                                ACC[key0][key1].append(float(acc_line.split(':')[-1].split('%')[0].strip()))
                                break
                        break
                    idx += 1
    ACC_all[key] = ACC

index = []
best, FPR95, BAUROC, BFPR = [], [], [], []
BUQC, BUQE, BTheta, Boodc, Bidc = [], [], [], [], []
df_all = []
val, test, FPR, AUROC = [], [], [], []

best_4d, best = [], []
UQ_C_4d = []
UQ_E_4d = []
Theta_4d = []
ood_c_4d = []
id_c_4d = []
AUROC_4d = []
loss_4d = []
FPR_4d, ACCS_4d = [], []
val_all, test_all = [[[], [], [], []], [[], [], [], []], [[], [], [], []]], [[[], [], [], []], [[], [], [], []],[[], [], [], []]]
loss_all = [[[], [], [], []], [[], [], [], []], [[], [], [], []]]
all_ACC_S, all_UQ_C, all_UQ_E, all_Theta, all_ood_c, all_id_c, all_AUROC, all_FPR = \
    [[[], [], [], []], [[], [], [], []], [[], [], [], []]], [[[], [], [], []], [[], [], [], []], [[], [], [], []]], \
    [[[], [], [], []], [[], [], [], []], [[], [], [], []]], [[[], [], [], []], [[], [], [], []], [[], [], [], []]], \
    [[[], [], [], []], [[], [], [], []], [[], [], [], []]], [[[], [], [], []], [[], [], [], []], [[], [], [], []]], \
    [[[], [], [], []], [[], [], [], []], [[], [], [], []]], [[[], [], [], []], [[], [], [], []], [[], [], [], []]]
all_loss = [[[], [], [], []], [[], [], [], []], [[], [], [], []]]

for idx, domain in enumerate(ACC_all.keys()):
    df = []
    key = []
    for seed in ACC_all[domain].keys():
        df_seed = []
        keylist = []
        for alpha in ACC_all[domain][seed].keys():
            c = []
            if alpha.split('_')[-1] == '1' : #and  alpha.split('_')[-3] == '0.5':
            # if True:
                tmp = ACC_all[domain][seed][alpha]
                keylist.append(seed + alpha)
                c.append(tmp)
                df_seed.append(c)
        df.append(df_seed)
        key.append(keylist)
    print(keylist)

    # get the final result (mean and std) of the 3 best model
    for k, tmp in enumerate(df):
        val, test, FPR, AUROC, loss = [], [], [], [], []
        val_acc, test_acc, loss_train = [], [], []
        test_AUROC0 = []
        test_FPR0 = []
        UQ_C, UQ_E, Theta, ood_c, id_c, ACC_S, loss_test = [], [], [], [], [], [], []
        test_ACC_S0, test_UQ_C0, test_UQ_E0, test_Theta0, test_ood_c0, test_id_c0 = [], [], [], [], [], []
        for i in range(len(tmp)):
            # print(key[k][i])
            val_acc.append(np.float32(tmp[i][0][0]))
            test_acc.append(np.float32(tmp[i][0][0]))
            AUROC.append(np.float32(tmp[i][0][2]))
            FPR.append(np.float32(tmp[i][0][1]))
            # ACC_S.append(np.float32(test_acc[-1] / 100 / (1 - (1 - FPR[-1]) * (1 - test_acc[-1] / 100))))
            ACC_S.append(np.float32(tmp[i][0][3]))
            UQ_C.append(np.float32(tmp[i][0][4]))
            UQ_E.append(np.float32(tmp[i][0][5]))
            loss_test.append(np.float32(tmp[i][0][6]))
            Theta.append(np.float32(tmp[i][0][7]))
            ood_c.append(np.float32(tmp[i][0][8]))
            id_c.append(np.float32(tmp[i][0][9]))
        val.append(val_acc)
        # loss.append(loss_train)
        test.append(test_acc)
        test_AUROC0.append(AUROC)
        test_FPR0.append(FPR)
        test_UQ_C0.append(UQ_C)
        test_UQ_E0.append(UQ_E)
        test_ACC_S0.append(ACC_S)
        test_Theta0.append(Theta)
        test_ood_c0.append(ood_c)
        test_id_c0.append(id_c)
        loss.append(loss_test)
        val_all[k][idx] = val
        all_loss[k][idx] = loss
        test_all[k][idx] = test
        all_AUROC[k][idx] = test_AUROC0
        all_FPR[k][idx] = test_FPR0
        all_UQ_C[k][idx] = test_UQ_C0
        all_UQ_E[k][idx] = test_UQ_E0
        all_ACC_S[k][idx] = test_ACC_S0
        all_Theta[k][idx] = test_Theta0
        all_ood_c[k][idx] = test_ood_c0
        all_id_c[k][idx] = test_id_c0

for seed in [0,1,2]:
    val = val_all[seed]
    test = test_all[seed]
    test_UQ_C = all_UQ_C[seed]
    test_UQ_E = all_UQ_E[seed]
    test_ACC_S = all_ACC_S[seed]
    test_Theta = all_Theta[seed]
    test_ood_c = all_ood_c[seed]
    test_id_c = all_id_c[seed]
    test_AUROC = all_AUROC[seed]
    test_FPR = all_FPR[seed]
    test_loss = all_loss[seed]
    val_best = 0
    for i in range(len(val[0][0])):
        val_4d = np.mean([val[d][0][i] for d in range(1)])
        if val_4d >= val_best:
            val_best = val_4d
            best_idx = i
            test_best = np.mean([test[d][0][i] for d in range(1)])
            test_UQC = np.mean([test_UQ_C[d][0][i] for d in range(1)])
            test_UQE = np.mean([test_UQ_E[d][0][i] for d in range(1)])
            test_Theta1 = np.mean([test_Theta[d][0][i] for d in range(1)])
            test_oodc = np.mean([test_ood_c[d][0][i] for d in range(1)])
            test_idc = np.mean([test_id_c[d][0][i] for d in range(1)])
            test_AUROC1 = np.mean([test_AUROC[d][0][i] for d in range(1)])
            test_FPR1 = np.mean([test_FPR[d][0][i] for d in range(1)])
            test_ACCS = np.mean([test_ACC_S[d][0][i] for d in range(1)])
            test_loss1 = np.mean([test_loss[d][0][i] for d in range(1)])
        print(key[seed][i][1:], np.mean([test[d][0][i] for d in range(1)]), np.mean([test_ACC_S[d][0][i] for d in range(1)]), val_best)
    best_4d.append(test_best)
    UQ_C_4d.append(test_UQC)
    UQ_E_4d.append(test_UQE)
    Theta_4d.append(test_Theta1)
    ood_c_4d.append(test_oodc)
    id_c_4d.append(test_idc)
    AUROC_4d.append(test_AUROC1)
    FPR_4d.append(test_FPR1)
    ACCS_4d.append(test_ACCS)
    loss_4d.append(test_loss1)
    print('hparams:',  test_best, val_best, test_ACCS, test_FPR1)
print(best_4d)
print('best_mean:', np.mean(best_4d), '.....', 'best_std:', np.std(best_4d))
print('best_uqc_result={}±{}'.format(np.mean(UQ_C_4d), np.std(UQ_C_4d)))
print('best_uqe_result={}±{}'.format(np.mean(UQ_E_4d), np.std(UQ_E_4d)))
print('best_theta_result={}±{}'.format(np.mean(Theta_4d), np.std(Theta_4d)))
print('best_oodc_result={}±{}'.format(np.mean(ood_c_4d), np.std(ood_c_4d)))
print('best_idc_result={}±{}'.format(np.mean(id_c_4d), np.std(id_c_4d)))
print('best_FPR_result={}±{}'.format(np.mean(FPR_4d), np.std(FPR_4d)))
print('best_AUROC_result={}±{}'.format(np.mean(AUROC_4d), np.std(AUROC_4d)))
print('best_ACCS_result={}±{}'.format(np.mean(ACCS_4d), np.std(ACCS_4d)))
print('best_loss_ood_result={}±{}'.format(np.mean(loss_4d), np.std(loss_4d)))



