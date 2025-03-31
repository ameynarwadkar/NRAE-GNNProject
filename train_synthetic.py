import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import argparse
import yaml
from omegaconf import OmegaConf
from loader import get_dataset, get_dataloader
from models import get_model
from PIL import Image  # For resizing frames

def run(cfg):
    # Setup seeds
    seed = cfg.get("seed", 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup device
    device = cfg.device
    
    # Setup Dataloader
    d_datasets = {}
    d_dataloaders = {}
    for key, dataloader_cfg in cfg["data"].items():
        d_datasets[key] = get_dataset(dataloader_cfg)
        d_dataloaders[key] = get_dataloader(dataloader_cfg)
    
    # Save Figure of Generated Data  
    training_data_fig = d_datasets['training'].visualize_data(
        d_datasets['training'].data,
        d_datasets['test'].data
    )
    
    graph_config = cfg["data"]["training"].get("graph", None)
    if graph_config is not None:
        graph_fig = d_datasets['training'].visualize_graph(
            d_datasets['training'].data,
            d_datasets['test'].data,
            d_datasets['training'].dist_mat_indices
        )
        
    # Setup Model
    model = get_model(cfg['model']).to(device)
    if graph_config is not None:
        model.dist_indices = d_datasets['training'].dist_mat_indices
    
    # Setup optimizer and scheduler
    params = {k: v for k, v in cfg['optimizer'].items() if k != "name"}
    optimizer = torch.optim.Adam(model.parameters(), **params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    list_of_images = []
    for epoch in range(cfg['training']['num_epochs']):
        training_loss = []
        if graph_config is not None:
            for x, x_nn in d_dataloaders['training']:
                train_dict = model.train_step(x.to(device), x_nn.to(device), optimizer)
                training_loss.append(train_dict["loss"])
        else:
            for x in d_dataloaders['training']:
                train_dict = model.train_step(x.to(device), optimizer)
                training_loss.append(train_dict["loss"])
        avg_loss = sum(training_loss) / len(training_loss)
        print(f"n_epoch: {epoch}, training_loss: {avg_loss:.6f}")
        
        if epoch > 0.8 * cfg['training']['num_epochs']:
            scheduler.step()
        
        # Capture snapshot every 40 epochs or during early training (< 30 epochs)
        if (epoch % 40 == 0) or (epoch < 30):
            image_array = model.synthetic_visualize(
                epoch,
                avg_loss,
                d_datasets['training'].data, 
                d_datasets['test'].data, 
                device
            )
            list_of_images.append(image_array)
            
    # Create an introductory frame with model name and title
    f = plt.figure()
    model_name = cfg['model']['arch'].upper()
    plt.text(0.5, 0.5, f'{model_name} Training', size=24, ha='center', va='center')
    plt.axis('off')
    f.canvas.draw()
    f_arr = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
    f_arr = f_arr.reshape(f.canvas.get_width_height()[::-1] + (3,))
    plt.close(f)
    
    # Assemble final GIF frames
    list_figs = [f_arr]*10 + [training_data_fig]*10 
    if graph_config is not None:
        list_figs += [graph_fig]*10
    list_figs += list_of_images
    list_figs += [list_of_images[-1]]*20

    # --- Ensure all frames have the same shape ---
    # Use the shape of the first frame as the target (e.g., (height, width, channels))
    fixed_shape = list_figs[0].shape  # For example, (H, W, 3)
    fixed_list_figs = []
    # Use LANCZOS as the resampling filter (new Pillow versions use Image.Resampling)
    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    for frame in list_figs:
        # Ensure frame is a numpy array of type uint8
        frame = np.array(frame, dtype=np.uint8)
        # If frame is grayscale, convert to RGB
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        # If channel number does not match, adjust:
        if frame.shape[-1] != fixed_shape[-1]:
            if frame.shape[-1] > fixed_shape[-1]:
                frame = frame[..., :fixed_shape[-1]]
            else:
                # Repeat channels if there are fewer than expected
                repeats = fixed_shape[-1] // frame.shape[-1]
                frame = np.repeat(frame, repeats, axis=-1)
        # Resize if the height or width is different from fixed_shape
        if frame.shape[:2] != fixed_shape[:2]:
            img_pil = Image.fromarray(frame)
            img_pil = img_pil.resize((fixed_shape[1], fixed_shape[0]), resample=resample_filter)
            frame = np.array(img_pil)
        fixed_list_figs.append(frame)
    
    imageio.mimsave(
        os.path.join(cfg['logdir'], f'{model_name}_TRAINING.gif'), 
        fixed_list_figs, duration=0.2)
    
def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if (val == "True") or (val == 'true'):
        return True
    if (val == "False") or (val == 'false'):
        return False
    try:
        return float(val)
    except ValueError:
        return val

def parse_unknown_args(l_args):
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args * 2]
        val = l_args[i_args * 2 + 1]
        assert "=" not in key, "optional arguments should be separated by space"
        kwargs[key.strip("-")] = parse_arg_type(val)
    return kwargs

def parse_nested_args(d_cmd_cfg):
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split(".")
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg

def save_yaml(filename, text):
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", default='results/')
    parser.add_argument("--device", default=0)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    if args.device == "cpu":
        cfg["device"] = "cpu"
    else:
        cfg["device"] = f"cuda:{args.device}"
    cfg['logdir'] = os.path.join(args.logdir, cfg['exp_name'])
    print(OmegaConf.to_yaml(cfg))

    # make save dir
    os.makedirs(cfg['logdir'], exist_ok=True)
    
    # copy config file
    copied_yml = os.path.join(cfg['logdir'], os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml}")

    run(cfg)
