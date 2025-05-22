import os
from PIL import Image
import argparse
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from dataclasses import dataclass
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


@dataclass
class TrainingConfig:
    image_size = 768
    train_batch_size = 4
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


def generate_images(model_path, test_dir, start_step, batch_size, epochs):
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    config = TrainingConfig()

    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    model.load_state_dict(torch.load(model_path))
    model.to('cuda')

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

    for _ in tqdm(range(epochs)):
        images = pipeline(
            batch_size=batch_size,
            # generator=torch.manual_seed(config.seed),
        ).images

        for image in images:
            image_grid = make_grid([image], rows=1, cols=1)
            image_grid.save(f"{test_dir}/{start_step}.png")
            start_step += 1


def main():
    parser = argparse.ArgumentParser(description='Generate images using a diffusion model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--start_step', type=int, default=0, help='Starting number for image filenames')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images to generate per batch')
    parser.add_argument('--epochs', type=int, default=1, help='Number of generation epochs to run')
    
    args = parser.parse_args()
    
    generate_images(
        model_path=args.model_path,
        test_dir=args.test_dir,
        start_step=args.start_step,
        batch_size=args.batch_size,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()


# python script.py --model_path path/to/model.pth --test_dir path/to/output --start_step 500 --batch_size 1 --epochs 450