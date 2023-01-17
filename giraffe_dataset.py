from torch.utils.data import Dataset
import random


class GiraffeDataset(Dataset):
    def __init__(self, attributes_list, dim_z=256, dataset_length=16 * 50, renderer=None, is_constant=False):
        """
        :param attributes_list:
        :param dim_z:
        :param dataset_length: Can be any number - dataset creating latent on the flight
        :param renderer:
        :param is_constant: if True, the dataset will return the same latent codes for each item. (For debugging)
        """
        self.attributes_list = attributes_list
        self.dim_z = dim_z
        self.dataset_length = int(dataset_length)
        self.renderer = renderer
        self.z_shape, self.z_app, self.gb_shape, self.gb_app = self.renderer.generator.get_latent_codes(1)
        self.z_shape.squeeze(0)
        self.z_app.squeeze(0)
        self.gb_shape.squeeze(0)
        self.gb_app.squeeze(0)

        self.is_constant = is_constant


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        """
        Get item for the training dataset. It returns some random latent codes and the corresponding attributes.
        :return:
        """
        attr = random.choice(self.attributes_list)

        if self.is_constant:
            return attr, (self.z_shape.squeeze(0), self.z_app.squeeze(0), self.gb_shape.squeeze(0), self.gb_app.squeeze(0))

        z_shape, z_app, gb_shape, gb_app = self.renderer.generator.get_latent_codes(1)
        return attr, (z_shape.squeeze(0), z_app.squeeze(0), gb_shape.squeeze(0), gb_app.squeeze(0))
