import torch
import torch.nn as nn
import cv2
from torch.nn import functional as F
from render import get_model_cars
from typing import List, Tuple


class model_z_a(nn.Module):
    def __init__(self, input_dim, output_dim=256, device='cuda'):
        """
        :param input_dim: should be dim_z_a + attributes_size
        :param output_dim: should be original dim_z_a
        """
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        ).to(device)

    def forward(self, x):
        return self.embedder(x)


class model_z_s(nn.Module):
    def __init__(self, input_dim, output_dim=256, device='cuda'):
        """
        :param input_dim: should be dim_z_s + attributes_size
        :param output_dim: should be original dim_z_s
        """
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        ).to(device)
        self.output_dim = output_dim

    def forward(self, x):
        # return x[:, :, :256] # If you want to only change appearance
        return self.embedder(x)


class Latent_z_to_w(nn.Module):
    def __init__(self, attributes_list, original_latent_dim=256, device='cuda'):
        super().__init__()
        self.attributes_list = attributes_list
        self.one_hot_attributes = torch.eye(len(attributes_list))
        self.z_app_model = model_z_a(original_latent_dim + len(attributes_list), original_latent_dim).to(device)
        self.z_shape_model = model_z_s(original_latent_dim + len(attributes_list), original_latent_dim).to(device)

    def forward(self, z_shape, z_app, attributes):
        """if attributes not in self.attributes_list:
            raise ValueError(f'Attribute {attributes} is not supported')
        if z_shape.shape[0] != z_app.shape[0]:
            raise ValueError(f'z_a and z_s should have the same batch size')"""

        attribute_encoding = self.one_hot_attributes[[self.attributes_list.index(att) for att in attributes]]
        attribute_encoding = attribute_encoding.to(z_shape.device)
        extended_z_app = torch.cat([z_app, attribute_encoding.unsqueeze(1)], dim=-1)
        extended_z_shape = torch.cat([z_shape, attribute_encoding.unsqueeze(1)], dim=-1)
        z_app = self.z_shape_model(extended_z_shape)
        z_shape = self.z_app_model(extended_z_app)
        return z_app.to('cuda'), z_shape.to('cuda')


class GiraffeGuided(nn.Module):
    def __init__(self, latent_z_to_w):
        super().__init__()
        _, self.renderer = get_model_cars()
        self.z_dim = self.renderer.generator.z_dim
        self.latent_z_to_w = latent_z_to_w

    def forward(self, attributes, latent_tuple):
        z_shape, z_app, gb_shape, gb_app = latent_tuple
        z_shape, z_app = self.latent_z_to_w(z_shape, z_app, attributes)
        batch_size = z_shape.shape[0]
        images = self.renderer.render_object_with_injected_latent_zs(batch_size, z_shape, z_app, gb_shape, gb_app)
        return images, (z_shape, z_app)

