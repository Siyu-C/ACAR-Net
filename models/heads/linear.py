import torch
import torch.nn as nn
import torchvision

__all__ = ['linear']


class LinearHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False):
        super(LinearHead, self).__init__()
        
        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)

        self.fc = nn.Linear(width, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    # data: features, rois
    # returns: outputs
    def forward(self, data):
        if not isinstance(data['features'], list):
            features = [data['features']]
        else:
            features = data['features']

        roi_features = []
        for f in features:
            sp = f.shape
            h, w = sp[3:]
            feats = nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, sp[1], h, w)

            rois = data['rois'].clone()
            rois[:, 1] = rois[:, 1] * w
            rois[:, 2] = rois[:, 2] * h
            rois[:, 3] = rois[:, 3] * w
            rois[:, 4] = rois[:, 4] * h
            rois = rois.detach()
            roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))
            roi_feats = self.roi_maxpool(roi_feats).view(-1, sp[1])

            roi_features.append(roi_feats)

        roi_features = torch.cat(roi_features, dim=1)
        if self.dp is not None:
            roi_features = self.dp(roi_features)
        outputs = self.fc(roi_features)

        return {'outputs': outputs}


def linear(**kwargs):
    model = LinearHead(**kwargs)
    return model