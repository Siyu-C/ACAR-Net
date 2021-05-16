import torch
import torch.nn as nn

from .backbones import AVA_backbone
from .necks import AVA_neck
from .heads import AVA_head


class AVA_model(nn.Module):
    def __init__(self, config):
        super(AVA_model, self).__init__()
        self.config = config
        
        self.backbone = AVA_backbone(config.backbone)
        self.neck = AVA_neck(config.neck)
        self.head = AVA_head(config.head)
        
    def forward(self, data, evaluate=False):
        if not evaluate: # train mode
            i_b = {'clips': data['clips']}
            o_b = self.backbone(i_b)

            i_n = {'aug_info': data['aug_info'], 'labels': data['labels'], 
                   'filenames': data['filenames'], 'mid_times': data['mid_times']}
            o_n = self.neck(i_n)

            if o_n['num_rois'] == 0:
                return {'outputs': None, 'targets': o_n['targets'], 
                        'num_rois': 0, 'filenames': o_n['filenames'], 
                        'mid_times': o_n['mid_times'], 'bboxes': o_n['bboxes']}
                
            i_h = {'features': o_b['features'], 'rois': o_n['rois'],
                   'num_rois': o_n['num_rois'], 'roi_ids': o_n['roi_ids'],
                   'sizes_before_padding': o_n['sizes_before_padding']}
            o_h = self.head(i_h)

            return {'outputs': o_h['outputs'], 'targets': o_n['targets'], 
                    'num_rois': o_n['num_rois'], 'filenames': o_n['filenames'], 
                    'mid_times': o_n['mid_times'], 'bboxes': o_n['bboxes']}
        
        # eval mode
        assert not self.training
        
        noaug_info = [{'crop_box': [0., 0., 1., 1.], 'flip': False, 'pad_ratio': [1., 1.]}] * len(data['labels'])
        i_n = {'aug_info': noaug_info, 'labels': data['labels'], 
               'filenames': data['filenames'], 'mid_times': data['mid_times']}
        o = self.neck(i_n)
        
        output_list = [None] * len(o['filenames'])
        cnt_list = [0] * len(o['filenames'])
        
        for no in range(len(data['clips'])):
            i_b = {'clips': data['clips'][no]}
            o_b = self.backbone(i_b)
            
            i_n = {'aug_info': data['aug_info'][no], 'labels': data['labels'], 
                   'filenames': data['filenames'], 'mid_times': data['mid_times']}
            o_n = self.neck(i_n)
            
            if o_n['num_rois'] == 0:
                continue
            ids = o_n['bbox_ids']
                
            i_h = {'features': o_b['features'], 'rois': o_n['rois'], 
                   'num_rois': o_n['num_rois'], 'roi_ids': o_n['roi_ids'],
                   'sizes_before_padding': o_n['sizes_before_padding']}
            o_h = self.head(i_h)
            
            outputs = o_h['outputs']
            for idx in range(o_n['num_rois']):
                if cnt_list[ids[idx]] == 0:
                    output_list[ids[idx]] = outputs[idx]
                else:
                    output_list[ids[idx]] += outputs[idx]
                cnt_list[ids[idx]] += 1
            
        num_rois, filenames, mid_times, bboxes, targets, outputs = 0, [], [], [], [], []
        for idx in range(len(o['filenames'])):
            if cnt_list[idx] == 0:
                continue
            num_rois += 1
            filenames.append(o['filenames'][idx])
            mid_times.append(o['mid_times'][idx])
            bboxes.append(o['bboxes'][idx])
            targets.append(o['targets'][idx])
            outputs.append(output_list[idx] / float(cnt_list[idx]))

        if num_rois == 0:
            return {'outputs': None, 'targets': None, 'num_rois': 0, 
                    'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes}
        
        final_outputs = torch.stack(outputs, dim=0)
        final_targets = torch.stack(targets, dim=0)
        return {'outputs': final_outputs, 'targets': final_targets, 'num_rois': num_rois, 
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes}

    def train(self, mode=True):
        super(AVA_model, self).train(mode)

        if mode and self.config.get('freeze_bn', False):
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.backbone.apply(set_bn_eval)
