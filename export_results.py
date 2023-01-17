import torch
import os
import argparse
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import matplotlib.pyplot as plt


def interpolation(_renderer, og, end, interval_nums=20):
    z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = torch.load(og)
    start = z_shape_obj, z_app_obj, z_shape_bg, z_app_bg
    z_shape_obj, z_app_obj = torch.load(end)
    end = z_shape_obj, z_app_obj, z_shape_bg, z_app_bg
    direction_z_shape = end[0] - start[0]
    direction_z_app = end[1] - start[1]
    interval_z_shape = direction_z_shape / interval_nums
    interval_z_app = direction_z_app / interval_nums
    gb_shape = start[2]
    gb_app = start[3]

    for i in range(interval_nums):
        z_shape = start[0] + interval_z_shape * i
        z_app = start[1] + interval_z_app * i
        image = _renderer.render_object_with_injected_latent_zs(1, z_shape, z_app, gb_shape, gb_app)
        save_image("./out/interpolation/{}.png".format(i), image[0])


def save_image(path, image):
    plt.imshow(image.detach().cpu().numpy().transpose(1, 2, 0))
    plt.savefig(path)


def render_results(og_object_path="", new_object_path="", mode=["object_rotation"], name="test"):
    z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = torch.load(og_object_path)
    z_shape_obj, z_app_obj, = torch.load(new_object_path)

    latent_tuple = z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    out = renderer.custom_render_full_visualization(
        render_dir,
        mode, latent_tuple, name)
    return out


if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    config_path = 'configs/256res/cars_256_pretrained.yaml'

    cfg = config.load_config(config_path, 'configs/default.yaml')
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    out_dir = cfg['training']['out_dir']
    render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    # Model
    model = config.get_model(cfg, device=device)
    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # Generator
    renderer = config.get_renderer(model, cfg, device=device)

    model.eval()
