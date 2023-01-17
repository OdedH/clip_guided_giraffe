import torch
import os
import argparse
from im2scene import config
from im2scene.checkpoints import CheckpointIO

def get_model_cars():
    """
    parser = argparse.ArgumentParser(
        description='Render images of a GIRAFFE model.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    """
    is_cuda = torch.cuda.is_available()
    config_path = 'configs/256res/cars_256_pretrained.yaml'
    cfg = config.load_config(config_path, 'configs/default.yaml')
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
    #out = renderer.render_full_visualization(
#        render_dir,
#        cfg['rendering']['render_program'])
    return model, renderer

if __name__ == '__main__':
    get_model_cars()