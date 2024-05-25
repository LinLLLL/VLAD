import os
import pickle
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

import numpy as np
import scipy.stats as stats
import pandas as pd
from PIL import Image, ImageFile

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden
from dassl.utils import read_json, write_json, check_isfile


@DATASET_REGISTRY.register()
class CelebA(DatasetBase):

    dataset_dir = "CelebA"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))  # /DATA
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        all_domain, all_class = [], []
        train, val, test = [], [], []
        self.train_data_path1 = os.path.join(self.dataset_dir, 'blond_split', 'tr_env1_df.pickle')
        self.train_data_path2 = os.path.join(self.dataset_dir, 'blond_split',  'tr_env2_df.pickle')
        self.val_data_path = os.path.join(self.dataset_dir, 'blond_split', 'te_env_df.pickle')
        self.test_data_path = os.path.join(self.dataset_dir, 'blond_split', 'te_env_df.pickle')

        for path in [self.train_data_path1, self.train_data_path2, self.val_data_path, self.test_data_path]:
            if os.path.exists(path):
                if 'tr' in path:
                    if len(train) == 0:
                        all_domain, all_class, train = self.read_pickle(path, all_domain, all_class)
                    else:
                        all_domain, all_class, train2 = self.read_pickle(path, all_domain, all_class)
                        # train.extend(train2)
                else:
                    all_domain, all_class, test = self.read_pickle(path, all_domain, all_class)


        num_shots = cfg.DATASET.NUM_SHOTS
#        train_samples = self.generate_train_samples_dataset(test, num_samples=20)
        train += train2
        train = self.generate_fewshot_dataset_balance(train, num_shots=num_shots) if num_shots > 4 else self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = self.generate_fewshot_dataset_balance(test, num_shots=8)
        test = [tmp for tmp in test if tmp not in val]
        super().__init__(train_x=train, val=val, test=test, train_samples=test, alldomain=all_domain)

    def read_pickle(self, filepath, all_domain, all_class):
        out = []
        dataframes = []
        with open(filepath, 'rb') as handle:
            dataframe = pickle.load(handle)
        target_id = 9
        images_path = os.path.join(self.dataset_dir, 'img_align_celeba')
        if 'tr' in filepath:
            images, labels = get_CelebA(pd.DataFrame(dataframe), images_path, target_id, transform=None, cdiv=0, ccor=1)
        else:
            images, labels = get_CelebA(pd.DataFrame(dataframe), images_path, target_id, transform=None, cdiv=0, ccor=0)
        for i in range(len(images)):
            img_path = os.path.join(images_path, images[i])
            if 'tr_env1' in filepath:  # ["unbalanced_1", "unbalanced_2", "balanced"]
                domainname = "male_female_unbalanced1"#'env1'
            elif 'tr_env2' in filepath:
                domainname = "male_female_unbalanced2"#'env2'
            else:
                domainname = "male_female_balanced"#'test'
            classname = 'blond_hair' if labels[i] == 1 else 'black_hair'
            self.img_dir = os.path.join(self.dataset_dir, img_path)
            if domainname not in all_domain:
                all_domain.append(domainname)
            if classname not in all_class:
                all_class.append(classname)
            domain_label = all_domain.index(domainname)
            label = all_class.index(classname)
            if check_isfile(self.img_dir):
                item = Datum(impath=self.img_dir, label=int(label), domain=int(domain_label), classname=classname, domainname=domainname)
                out.append(item)
        return all_domain, all_class, out

        
def get_CelebA(dataframe, folder_dir, target_id, transform=None, cdiv=0, ccor=0):
    file_names = dataframe.index
    targets = np.concatenate(dataframe.labels.values).astype(int)
    gender_id = 20

    target_idx0 = np.where(targets[:, target_id] == 0)[0]  # nontarget
    target_idx1 = np.where(targets[:, target_id] == 1)[0]  # target
    gender_idx0 = np.where(targets[:, gender_id] == 0)[0]  # 0 females
    gender_idx1 = np.where(targets[:, gender_id] == 1)[0] # 1-males
    nontarget_males = list(set(gender_idx1) & set(target_idx0))
    nontarget_females = list(set(gender_idx0) & set(target_idx0))
    target_males = list(set(gender_idx1) & set(target_idx1))
    target_females = list(set(gender_idx0) & set(target_idx1))

    u1 = len(nontarget_males) - int((1 - ccor) * (len(nontarget_males) - len(nontarget_females)))  # len(nontarget_males)
    u2 = len(target_females) - int((1 - ccor) * (len(target_females) - len(target_males)))  # len(target_females)
    selected_idx = nontarget_males[:u1] + nontarget_females + target_males + target_females[:u2]
    targets = targets[selected_idx]
    file_names = file_names[selected_idx]

    target_idx0 = np.where(targets[:, target_id] == 0)[0]
    target_idx1 = np.where(targets[:, target_id] == 1)[0]
    gender_idx0 = np.where(targets[:, gender_id] == 0)[0]
    gender_idx1 = np.where(targets[:, gender_id] == 1)[0]
    nontarget_males = list(set(gender_idx1) & set(target_idx0))
    nontarget_females = list(set(gender_idx0) & set(target_idx0))
    target_males = list(set(gender_idx1) & set(target_idx1))
    target_females = list(set(gender_idx0) & set(target_idx1))

    selected_idx = nontarget_males + nontarget_females[:int(len(nontarget_females) * (1 - cdiv))] + target_males + target_females[:int(len(target_females) * (1 - cdiv))]
    targets = targets[selected_idx]
    file_names = file_names[selected_idx]

    target_idx0 = np.where(targets[:, target_id] == 0)[0]
    target_idx1 = np.where(targets[:, target_id] == 1)[0]
    gender_idx0 = np.where(targets[:, gender_id] == 0)[0]
    gender_idx1 = np.where(targets[:, gender_id] == 1)[0]
    nontarget_males = list(set(gender_idx1) & set(target_idx0))
    nontarget_females = list(set(gender_idx0) & set(target_idx0))
    target_males = list(set(gender_idx1) & set(target_idx1))
    target_females = list(set(gender_idx0) & set(target_idx1))
    print(len(nontarget_males), len(nontarget_females), len(target_males), len(target_females))
#    11671 462 462 11671
#    11209 924 924 11209
#    362 362 362 362
#    362 362 362 362

    targets = targets[:, target_id]

    image = file_names
    label = targets
    return image, label

