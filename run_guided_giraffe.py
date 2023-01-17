import wandb
import torch
import torchvision
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch import optim

from giraffe_guided import model_z_a, model_z_s, Latent_z_to_w, GiraffeGuided
import clip
import os
from torch.utils.data import DataLoader
from giraffe_dataset import GiraffeDataset
import cv2
from clip.clip import BICUBIC
from torch.optim import lr_scheduler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import matplotlib.pyplot as plt
import os


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--attribute', default="red", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--clip_model', type=str, default="ViT-B/32")
    parser.add_argument('--clip_embedding_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=32)
    parser.add_argument('--save_path', type=str, default="../checkpoints/straightforward_projection")
    parser.add_argument('--save_dir', type=str, default="../saves")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_coeff', type=float, default=1)
    parser.add_argument('--l2_coeff', type=float, default=0.1)
    parser.add_argument('--images_folder_name', type=str, default="out")
    parser.add_argument('--original_latent_dim', type=int, default=256)
    parser.add_argument('--number_epochs', type=int, default=1e6)
    parser.add_argument('--dataset_length', type=int, default=1000)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class GiraffeLoss:
    def __init__(self, clip_preprocess, device, clip_model, prompt, attributes):
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.clip_model = clip_model
        self.prompt = prompt
        self.other_attributes = attributes

    def _get_clip_loss(self, images, attributes):
        prompts = [self.prompt.format(attribute) for attribute in attributes]
        bad_prompts = [self.prompt.format(attribute) for attribute in self.other_attributes]
        inserted_images = self.clip_preprocess(images).to(self.device)
        prompt_tokens = clip.tokenize(prompts).to(self.device)
        bad_prompt_tokens = clip.tokenize(bad_prompts).to(self.device)
        bad_scores = torch.sum(self.clip_model(inserted_images, bad_prompt_tokens)[0], axis=1)
        good_scores = (len(self.other_attributes) + 1) * torch.diag(self.clip_model(inserted_images, prompt_tokens)[0])
        final = good_scores - bad_scores
        loss = -1 * torch.sum(final) / len(images)
        return loss

    def _l2_loss(self, og_z_app, og_z_shape, z_app, z_shape):
        app_dist = torch.norm(og_z_app - z_app)
        shape_dist = torch.norm(og_z_shape - z_shape)
        return app_dist + shape_dist

    def get_loss(self, image, attributes,
                 og_z_app, og_z_shape, z_app, z_shape,
                 coeff_clip_loss, coeff_l2_dist):
        clip_loss = self._get_clip_loss(image, attributes)
        l2_loss = self._l2_loss(og_z_app, og_z_shape, z_app, z_shape)
        return clip_loss * coeff_clip_loss + l2_loss * coeff_l2_dist


def save_instance(dataloader, model, args, epoch=0, batch=0, num_images_to_print=10):
    dir_location = rf'./{args.images_folder_name}/epoch_{epoch}'
    if not os.path.exists(dir_location):
        os.makedirs(dir_location)
    for batch_idx, (attributes, og_latent_tuple) in enumerate(dataloader):
        images, (z_shape, z_app) = model(attributes, og_latent_tuple)
        for i, image in enumerate(images):
            save_image(args, attributes, batch, epoch, i, images)
            save_latent_vectors(args, og_latent_tuple,(z_shape,z_app), batch, epoch, i)
            if i < num_images_to_print:
                break
        break


def save_image(args, attributes, batch, epoch, i, images):
    plt.imshow(images[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.savefig(rf'./{args.images_folder_name}/epoch_{epoch}/batch_{batch}_{attributes[i]}_{i}_.png')

def save_latent_vectors(args, og_latent_tuple, new_z_tuple, batch, epoch, i):
    torch.save(og_latent_tuple, rf'./{args.images_folder_name}/epoch_{epoch}/batch_{batch}_{i}_og_latent_tuple.pt')
    torch.save(new_z_tuple, rf'./{args.images_folder_name}/epoch_{epoch}/batch_{batch}_{i}_new_latent_tuple.pt')

def train_loop(dataloader, model, loss_fn, optimizer, epoch, args):
    size = len(dataloader.dataset)
    for batch_idx, (attributes, latent_tuple) in enumerate(dataloader):
        # Compute prediction and loss
        images, (z_shape, z_app) = model(attributes, latent_tuple)
        if epoch % 1 == 0 and batch_idx % 10 == 0:
            save_instance(dataloader, model, args, epoch, batch_idx)
        loss = loss_fn(images, attributes, latent_tuple[0], latent_tuple[1],
                       z_app, z_shape,
                       args.clip_coeff, args.l2_coeff)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * len(attributes)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


def cli_main():
    # ------------
    # args & Setup
    # ------------
    pl.seed_everything(546)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    args = get_args()
    # ------------
    # CLIP
    # ------------
    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model = clip_model.eval()
    preprocess = _transform(clip_model.visual.input_resolution)
    # ------------
    # Models
    # ------------
    attributes = [args.attribute]
    latent_z_to_w = Latent_z_to_w(attributes, args.original_latent_dim).to(device)
    giraffe_guided = GiraffeGuided(latent_z_to_w).to(device)
    # ------------
    # Logging
    # ------------
    wandb.init(project="Giraffe", name=f"general")
    # ------------
    # Data
    # ------------
    dataset = GiraffeDataset(attributes, giraffe_guided, dataset_length=args.dataset_length,
                             renderer=giraffe_guided.renderer, is_constant=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # ------------
    # Training
    # ------------
    prompt = "an image of a {} car"
    loss_calculator = GiraffeLoss(preprocess, device, clip_model, prompt=prompt, attributes=attributes)
    non_frozen_params = giraffe_guided.latent_z_to_w.parameters()
    optimizer = optim.Adam(non_frozen_params, lr=args.lr)
    best_loss = 0
    for t in range(int(args.number_epochs)):
        loss = train_loop(dataloader, giraffe_guided, loss_calculator.get_loss, optimizer, t, args)
        if loss < best_loss:
            best_loss = loss
            torch.save(latent_z_to_w.state_dict(), rf'./best_model.pt')

if __name__ == "__main__":
    cli_main()
