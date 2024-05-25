import numpy as np
import os.path as osp
from torch.nn import functional as F
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self._UQ_c = []
        self._UQ_e = []
        self.loss_test = []
        self.confidence_gt, self.detect, self.res_all = [], [], []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._UQ_c = []
        self._UQ_e = []
        self.loss_test = []
        self.confidence_gt, self.detect, self.res_all = [], [], []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        mo = mo[0] if isinstance(mo, list) else mo
        # mo = mo[0] if len(mo) ==5 else mo
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)
                
    def UQ(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        mo = mo[0] if isinstance(mo, list) else mo
        # mo = mo[0] if len(mo) ==5 else mo
        pred = mo.max(1)[1]
        softmax_logits = torch.nn.functional.softmax(mo, 1)
        
        res = softmax_logits.max(1)[0]
        matches = pred.eq(gt).float()
        error_idx = torch.where(matches==0)[0].cpu().numpy().tolist()
        correct_idx = torch.where(matches==1)[0].cpu().numpy().tolist()

        self._UQ_e.extend(res[error_idx].cpu().numpy().tolist())
        self._UQ_c.extend(res[correct_idx].cpu().numpy().tolist())
        loss_test = F.cross_entropy(mo[error_idx], gt[error_idx])
        self.loss_test.append(loss_test.item())
        self.confidence_gt.extend([1 if i in correct_idx else 0 for i in range(res.shape[0])])
        self.res_all.extend(res.cpu().numpy().tolist())
        
    def evaluate(self, theta_test=None):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["UQ_c"] = np.sum(self._UQ_c)/self._correct if self._correct > 0 else 0
        results["UQ_e"] = np.sum(self._UQ_e)/(self._total - self._correct)
        results["loss_test"] = np.sum(self.loss_test)/(self._total - self._correct)

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%"
        )
        print('results["UQ_c"]:', np.sum(self._UQ_c)/self._correct)
        UQ_e = np.sum(self._UQ_e)/(self._total - self._correct) if self._total - self._correct > 0 else 0
        print('results["UQ_e"]:', UQ_e)
        print('results["loss_test"]', np.sum(self.loss_test)/self._total)
        
        theta = np.percentile(np.array(self._UQ_c), 5) if np.sum(self._UQ_c) > 0 else 0
        if theta_test is not None:
            print('threshold @ TPR95:', theta, theta_test)
            theta = theta_test
        else:
            print('threshold @ TPR95:', theta)
        results["threshold"] = theta
        FPR95 = len(np.where(np.array(self._UQ_e) > theta)[0].tolist())
        max_UQ_e = np.max(self._UQ_e) if len(self._UQ_e) > 0 else 0
        print('max confidence score of ood datasets', max_UQ_e)
        max_score_id = np.max(self._UQ_c) if np.sum(self._UQ_c) > 0 else 0
        print('max confidence score of id datasets', max_score_id)
        FPR95 = FPR95/len(self._UQ_e) if len(self._UQ_e) > 0 else 1
        print('FPR95:', FPR95)
#        print('self.res.shape', np.array(self.res_all).shape)
        self.detect = np.where(np.array(self.res_all) > theta, 1, 0)
        if len(self._UQ_e) > 0 and len(self._UQ_c) > 0:
            AUROC = roc_auc_score(np.array(self.confidence_gt), self.detect)
        else:
            AUROC = None
        print('AUROC:', AUROC)
        print('ACCURACY:', -1)
        
        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    "* class: {} ({})\t"
                    "total: {:,}\t"
                    "correct: {:,}\t"
                    "acc: {:.2f}%".format(
                        label, classname, total, correct, acc
                    )
                )
            mean_acc = np.mean(accs)
            print("* average: {:.2f}%".format(mean_acc))

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results
