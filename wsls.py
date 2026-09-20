import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter, OrderedDict
from sklearn.preprocessing import normalize
import time
import pickle
from utils import fliplr


class CMA(nn.Module):
    """
    Dynamic switching:
    Use Version 1 (no confidence matching) for the first 70 epochs (Stage 2),
    Use Version 2 (high-confidence matching) after 70 epochs.
    """

    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device('cuda:0')
        self.not_saved = True
        self.num_classes = args.num_classes
        self.T = args.temperature if hasattr(args, 'temperature') else 0.8
        self.sigma = args.sigma
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, 2048, device=self.device))
        self.register_buffer('ir_memory', torch.zeros(self.num_classes, 2048, device=self.device))
        self.conf_thresh = args.match_conf_thresh if hasattr(args, 'match_conf_thresh') else 0.5
        self.memory_weight = defaultdict(float)
        self.args = args

    @torch.no_grad()
    def save(self, vis, ir, rgb_ids, ir_ids, rgb_idx, ir_idx, mode, rgb_features=None, ir_features=None):
        """Save scores/features and initialize memory bank with mean features"""
        self.mode = mode
        self.not_saved = False
        if self.mode not in ['scores', 'features']:
            raise ValueError('invalid mode!')
        elif self.mode == 'scores':
            vis = torch.nn.functional.softmax(self.T * vis, dim=1)
            ir = torch.nn.functional.softmax(self.T * ir, dim=1)

        # Memory bank initialization (same logic for both versions; Version 2 adds weight recording)
        if rgb_features is not None and ir_features is not None:
            label_set = torch.unique(rgb_ids)
            for label in label_set:
                rgb_mask = (rgb_ids == label)
                ir_mask = (ir_ids == label)
                if rgb_mask.any():
                    rgb_selected = rgb_features[rgb_mask]
                    self.vis_memory[label] = rgb_selected.mean(dim=0)
                    self.memory_weight[f'vis_{label}'] = 1.0  # Version 2: weight initialization
                if ir_mask.any():
                    ir_selected = ir_features[ir_mask]
                    self.ir_memory[label] = ir_selected.mean(dim=0)
                    self.memory_weight[f'ir_{label}'] = 1.0  # Version 2: weight initialization

        vis = vis.detach().cpu().numpy()
        ir = ir.detach().cpu().numpy()
        rgb_ids, ir_ids = rgb_ids.cpu(), ir_ids.cpu()
        self.vis, self.ir = vis, ir
        self.rgb_ids, self.ir_ids = rgb_ids, ir_ids
        self.rgb_idx, self.ir_idx = rgb_idx, ir_idx

    @torch.no_grad()
    def update(self, rgb_feats, ir_feats, rgb_labels, ir_labels):
        """Update memory bank with moving average"""
        rgb_set = torch.unique(rgb_labels)
        ir_set = torch.unique(ir_labels)

        for i in rgb_set:
            rgb_mask = (rgb_labels == i)
            selected_rgb = rgb_feats[rgb_mask].mean(dim=0)
            self.vis_memory[i] = (1 - self.sigma) * self.vis_memory[i] + self.sigma * selected_rgb

        for i in ir_set:
            ir_mask = (ir_labels == i)
            selected_ir = ir_feats[ir_mask].mean(dim=0)
            self.ir_memory[i] = (1 - self.sigma) * self.ir_memory[i] + self.sigma * selected_ir

    def get_label(self, epoch=None, conf_thresh=0.5, is_quantity_priority=False):
        """Route to matching strategy: quantity-first or confidence-first filtering"""
        if self.not_saved:
            return {}, {}, {}, {} if epoch > 100 else {}, {}

        if not is_quantity_priority:
            print('get match labels (quantity priority)')
            v2i_dict, i2v_dict = {}, {}
            if self.mode == 'features':
                dists = np.matmul(self.vis, self.ir.T)
                v2i_dict, i2v_dict = self._get_label_v1(dists, 'dist')
            elif self.mode == 'scores':
                v2i_dict, _ = self._get_label_v1(self.vis, 'rgb')
                i2v_dict, _ = self._get_label_v1(self.ir, 'ir')
            self.v2i = v2i_dict
            self.i2v = i2v_dict
            return v2i_dict, i2v_dict, {}, {}
        else:
            print('get match labels with confidence filter (quality priority)')
            v2i_dict = {}
            i2v_dict = {}
            v2i_conf = {}
            i2v_conf = {}
            if self.mode == 'features':
                dists = np.matmul(self.vis, self.ir.T)
                v2i_dict, i2v_dict, v2i_conf, i2v_conf = self._get_label_with_conf(
                    dists, 'dist', conf_thresh, epoch=epoch)
            elif self.mode == 'scores':
                rgb_v2i, _, rgb_v2i_conf, _ = self._get_label_with_conf(
                    self.vis, 'rgb', conf_thresh, epoch=epoch)
                ir_i2v, _, ir_i2v_conf, _ = self._get_label_with_conf(
                    self.ir, 'ir', conf_thresh, epoch=epoch)
                v2i_dict = rgb_v2i
                i2v_dict = ir_i2v
                v2i_conf = rgb_v2i_conf
                i2v_conf = ir_i2v_conf
            self.v2i = v2i_dict
            self.i2v = i2v_dict
            return v2i_dict, i2v_dict, v2i_conf, i2v_conf

    def _get_label_v1(self, dists, mode):
        """Version 1: Coarse matching without confidence filtering"""
        sample_rate = 1
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)
        idx1, idx2 = sorted_2d[0], sorted_2d[1]
        dists = dists[idx1, idx2]
        idx_length = int(np.ceil(sample_rate * dists.shape[0] / self.num_classes))
        dists = dists[:idx_length]

        if mode == 'dist':
            convert_label = [(i, j) for i, j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],
                                                    np.array(self.ir_ids)[idx2[:idx_length]])]
        elif mode == 'rgb':
            convert_label = [(i, j) for i, j in zip(np.array(self.rgb_ids)[idx1[:idx_length]],
                                                    idx2[:idx_length])]
        elif mode == 'ir':
            convert_label = [(i, j) for i, j in zip(np.array(self.ir_ids)[idx1[:idx_length]],
                                                    idx2[:idx_length])]
        else:
            raise AttributeError('invalid mode!')

        convert_label_cnt = Counter(convert_label)
        convert_label_cnt_sorted = sorted(convert_label_cnt.items(), key=lambda x: x[1], reverse=True)
        length = len(convert_label_cnt_sorted)
        in_rgb_label = []
        in_ir_label = []
        v2i = OrderedDict()
        i2v = OrderedDict()

        length_ratio = 1
        for i in range(int(length * length_ratio)):
            key = convert_label_cnt_sorted[i][0]
            if key[0] in in_rgb_label or key[1] in in_ir_label:
                continue
            in_rgb_label.append(key[0])
            in_ir_label.append(key[1])
            v2i[key[0]] = key[1]
            i2v[key[1]] = key[0]

        return v2i, i2v

    def _get_label_with_conf(self, dists, mode, conf_thresh=0.5, epoch=None):
        """Version 2: High-confidence matching with confidence threshold filtering"""
        sample_rate = 1.0
        dists_shape = dists.shape
        sorted_1d = np.argsort(dists, axis=None)[::-1]
        sorted_2d = np.unravel_index(sorted_1d, dists_shape)
        idx1, idx2 = sorted_2d[0], sorted_2d[1]
        dist_values = dists[idx1, idx2]
        idx_length = int(np.ceil(max(sample_rate * dists_shape[0] / self.num_classes, dists_shape[0] * 2)))
        idx_length = min(idx_length, len(idx1))

        convert_label = []
        convert_conf = []
        batch_max_dist = dist_values[:idx_length].max()
        batch_min_dist = dist_values[:idx_length].min()
        dist_range = batch_max_dist - batch_min_dist + 1e-8

        for i in range(idx_length):
            if mode == 'dist':
                r_id = np.array(self.rgb_ids)[idx1[i]]
                i_id = np.array(self.ir_ids)[idx2[i]]
            elif mode == 'rgb':
                r_id = np.array(self.rgb_ids)[idx1[i]]
                i_id = idx2[i]
            elif mode == 'ir':
                r_id = np.array(self.ir_ids)[idx1[i]]
                i_id = idx2[i]
            else:
                raise AttributeError('invalid mode!')
            conf = (dist_values[i] - batch_min_dist) / dist_range
            convert_label.append((r_id, i_id))
            convert_conf.append(conf)

        label_cnt = Counter()
        conf_dict = defaultdict(float)
        for (r, i), conf in zip(convert_label, convert_conf):
            if conf < conf_thresh:
                continue
            if (r, i) not in conf_dict or conf > conf_dict[(r, i)]:
                conf_dict[(r, i)] = conf
            label_cnt[(r, i)] += 1

        sorted_items = sorted(label_cnt.items(),
                              key=lambda x: (conf_dict[x[0]] * 10 + x[1]),
                              reverse=True)
        in_rgb = set()
        in_ir = set()
        v2i_dict = {}
        i2v_dict = {}
        v2i_conf = {}
        i2v_conf = {}

        for (r, i), cnt in sorted_items:
            if r in in_rgb or i in in_ir:
                continue
            # Record matching pairs and confidence scores
            conf = conf_dict[(r, i)]
            v2i_dict[r] = i
            i2v_dict[i] = r
            v2i_conf[r] = conf
            i2v_conf[i] = conf
            in_rgb.add(r)
            in_ir.add(i)

        if len(v2i_dict) < max(10, self.num_classes * 0.1):
            for (r, i), cnt in sorted_items:
                if r not in in_rgb and i not in in_ir:
                    conf = conf_dict[(r, i)]
                    v2i_dict[r] = i
                    i2v_dict[i] = r
                    v2i_conf[r] = conf
                    i2v_conf[i] = conf
                    in_rgb.add(r)
                    in_ir.add(i)
                    if len(v2i_dict) >= max(10, self.num_classes * 0.1):
                        break

        return v2i_dict, i2v_dict, v2i_conf, i2v_conf

    def extract(self, args, model, dataset, stable_flag=False, enable_phase1=True):
        """Extract RGB and IR features and save them to memory bank"""
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader()
        with torch.no_grad():
            rgb_features, rgb_labels, rgb_gt, r2i_cls, rgb_idx = self._extract_feature(
                model, rgb_loader, 'rgb', stable_flag, enable_phase1=enable_phase1)
            ir_features, ir_labels, ir_gt, i2r_cls, ir_idx = self._extract_feature(
                model, ir_loader, 'ir', stable_flag, enable_phase1=enable_phase1)
        self.save(r2i_cls, i2r_cls, rgb_labels, ir_labels, rgb_idx, ir_idx, 'scores', rgb_features, ir_features)

    def _extract_feature(self, model, loader, modal, stable_flag=False, enable_phase1=True):
        """Extract features and classification scores for one modality (RGB/IR)"""
        print(f'Extracting {modal} features{" with denoising..." if stable_flag else "..."}')
        saved_features, saved_labels, saved_cls = None, None, None
        saved_gts, saved_idx = None, None

        for imgs_list, infos in loader:
            labels = infos[:, 1]
            idx = infos[:, 0]
            gts = infos[:, -1].to(model.device)

            # Data augmentation processing
            if isinstance(imgs_list, list):
                ori_imgs, ca_imgs = imgs_list[0], imgs_list[1]
                if len(ori_imgs.shape) < 4:
                    ori_imgs = ori_imgs.unsqueeze(0)
                    ca_imgs = ca_imgs.unsqueeze(0)
                imgs = torch.cat((ori_imgs, ca_imgs), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                idx = torch.cat((idx, idx), dim=0)
                gts = torch.cat((gts, gts), dim=0).to(model.device)
            else:
                imgs = imgs_list

            imgs, labels, idx = imgs.to(model.device), labels.to(model.device), idx.to(model.device)
            bn_features = []

            if modal == 'rgb':
                _, bn_features = model.model(x1=imgs, enable_phase1=enable_phase1)
                if hasattr(model, 'feature_diffusion') and stable_flag:
                    bn_features = model.feature_diffusion.get_fused_feat(bn_features)
                cls, _ = model.classifier2(bn_features)
            elif modal == 'ir':
                _, bn_features = model.model(x2=imgs, enable_phase1=enable_phase1)
                if hasattr(model, 'feature_diffusion') and stable_flag:
                    bn_features = model.feature_diffusion.get_fused_feat(bn_features)
                cls, _ = model.classifier1(bn_features)
            else:
                raise ValueError(f'Invalid modal: {modal}')

            cls = cls.detach().cpu()
            bn_features = bn_features.detach().cpu()
            labels = labels.cpu()
            idx = idx.cpu()
            gts = gts.cpu()

            if saved_features is None:
                saved_features = bn_features
                saved_labels = labels
                saved_cls = cls
                saved_gts = gts
                saved_idx = idx
            else:
                saved_features = torch.cat((saved_features, bn_features), dim=0)
                saved_labels = torch.cat((saved_labels, labels), dim=0)
                saved_cls = torch.cat((saved_cls, cls), dim=0)
                saved_gts = torch.cat((saved_gts, gts), dim=0)
                saved_idx = torch.cat((saved_idx, idx), dim=0)

        assert len(saved_features) == len(saved_labels) == len(saved_cls) == len(saved_idx), \
            f"Feature extraction dimension mismatch"
        print(f'Extracted {modal} features shape: {saved_features.shape}')
        return saved_features, saved_labels, saved_gts, saved_cls, saved_idx