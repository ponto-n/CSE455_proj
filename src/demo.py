import argparse
import os

from PIL import Image
from transformers import AutoConfig, AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import torch
import numpy as np

from model import ResNetForImageRotation

def main(args):
    config = AutoConfig.from_pretrained(args.model, num_labels=1)
    model = ResNetForImageRotation.from_pretrained(
        args.model, config=config
    ).eval()
    
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    _transforms = Compose(
        [
            RandomResizedCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    _transforms2 = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Loaded model, running evaluation...")
    for i, file in enumerate(args.files):
        image = Image.open(file).convert("RGB")
        if args.crop_strategy == "option1":
            # First Option
            batch = torch.stack([_transforms(image.copy()) for _ in range(args.bs)]).to(device)
        elif args.crop_strategy == "option2":
            # Second Option
            batch = _transforms2(image.copy()).unsqueeze(0).to(device)
        else:
            assert False, f"Unknown crop strategy [{args.crop_strategy}]"
        # TODO median? Evaluate which strategy is best?
        output = model(batch)
        angle = output.logits.detach().cpu().numpy().mean()

        output_filename = os.path.join(args.output, os.path.basename(file))

        correction_angle = (-angle / (2 * np.pi) * 360)
        corrected_image = image.rotate(correction_angle, expand=True)
        corrected_image.save(output_filename)
        print(f"Processed image {i+1}/{len(args.files)} named {file}")
        print(f"Predicted correction angle: {correction_angle:.2f} degrees CW")



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--files", nargs="+", required=True, help="Filenames of image files to process")
    parser.add_argument("--output", help="Output directory to put rotated images")
    parser.add_argument("--bs", type=int, default=32, help="Number of times for the model to look at each file")
    parser.add_argument("--crop_strategy", default='option1', help="Method for cropping the image for the model")

    main(parser.parse_args())
