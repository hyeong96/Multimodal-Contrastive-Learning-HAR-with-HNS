import torch
import os
from models.S3D import S3D
from models.IMUEncoder import Encoder
from torch import nn


class MainEncoder(nn.Module):
    def __init__(self, input_size, s3d_weight_path, imu_weight_path, output_size):
        super(MainEncoder, self).__init__()
        self.video_encoder = S3D(27)
        self.imu_encoder = Encoder(6, output_size)
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 27))
        #load kinetics400 weights
        #self.video_encoder.load_state_dict(new_state_dict, strict=False)
        print(os.path.abspath(__file__))
        self.video_encoder.load_state_dict(torch.load(s3d_weight_path))
        self.imu_encoder.load_state_dict(torch.load(imu_weight_path))
        self.freeze_layers(self.video_encoder, ['fc2', 'fc', '15', '14'])
        self.video_encoder.fc2 = nn.Sequential()
        self.imu_encoder.fc = nn.Sequential()


    def freeze_layers(self, model, layers):
      for param in model.parameters():
        param.requires_grad = False

      for name, param in model.named_parameters():
        if param.requires_grad is False and any(x in name for x in layers):
          param.requires_grad = True

    def forward(self, video_feat, imu_feat):
        feat_vid = self.video_encoder(video_feat)
        feat_imu = self.imu_encoder(imu_feat)
        logits = self.linear_stack(torch.concat((feat_vid, feat_imu), dim=1))
        return logits