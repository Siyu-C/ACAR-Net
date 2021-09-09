# Model Zoo

Trained models are released in the Google drive folder [`model_zoo`](https://drive.google.com/drive/folders/1lTej6EaBjl7bSAeIvcXyOD0qvXXmrST3?usp=sharing), and should be downloaded to this local `model_zoo` folder by default for evaluation.

Name | Train Config | Eval Config | Orig mAP | Re-eval mAP
--- | :---: | :---: | :---: | :---:
[`AVA_SLOWFAST_R50_ACAR_HR2O.pth.tar`](https://drive.google.com/file/d/12vGZoWElxB-Lhn_sGz1aRMqAacexgKtm/view?usp=sharing) | [config](https://github.com/Siyu-C/ACAR-Net/blob/master/configs/AVA/SLOWFAST_R50_ACAR_HR2O.yaml) | [config](https://github.com/Siyu-C/ACAR-Net/blob/master/configs/AVA/eval_SLOWFAST_R50_ACAR_HR2O.yaml) | 27.83 | 27.74
[`AVA_SLOWFAST_R101_ACAR_HR2O_DEPTH1.pth.tar`](https://drive.google.com/file/d/1d9UdvwS0HR-h84j_z4JbRWwEbITSt_FN/view?usp=sharing) | -- | [config](https://github.com/Siyu-C/ACAR-Net/blob/master/configs/AVA/eval_SLOWFAST_R101_ACAR_HR2O_DEPTH1.yaml) | 31.69 | 31.72

Additional notes:
- Original mAPs (the column "Orig mAP") are results of evaluation with out-dated dependencies (torch <=1.3 and ffmpeg 3.X).
- Re-evaluated mAPs (the column "Re-eval mAP") are obtained with more recent packages (torch 1.8.1 and ffmpeg 4.4).
