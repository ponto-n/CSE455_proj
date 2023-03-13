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
import math

from model import ResNetForImageRotation


def main(args):
    imageRotationDemo = ImageRotationDemo(args.model,
                                          args.crop_strategy,
                                          args.bs)
    imageRotationDemo.fix_rotation(args.files, args.output)


class ImageRotationDemo:

    def __init__(self, model_dir, crop_strategy="option1", bs=32) -> None:
        self._config = AutoConfig.from_pretrained(model_dir, num_labels=1)
        self._model = ResNetForImageRotation.from_pretrained(
            model_dir, config=self._config
        ).eval()

        image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/resnet-50")
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"],
                    image_processor.size["width"])
        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )
        self._transforms = Compose(
            [
                RandomResizedCrop(size),
                ToTensor(),
                normalize,
            ]
        )
        self._transforms2 = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(self._device)
        self._crop_strategy = crop_strategy
        self._bs = bs
        print(f"Loaded model from directory {model_dir}")

    def fix_rotation(self, image_paths: list, output_dir: str, v=False) -> dict:
        '''
        Use the model to predict the rotation of the image, then correct the 
        rotation and save the corrected image to the output directory.
        Args:
            image_paths: list of paths to images to fix their rotation
            output_dir: directory to save the corrected images
            v: print verbose output about rotation correction
        Returns:
            dictionary mapping the file name to the predicted degrees to fix
        '''
        predictions = {}
        for i, img_path in enumerate(image_paths):
            image = Image.open(img_path).convert("RGB")
            angle = self.predict_rotation(image)

            # Correct the rotation of the image
            correction_angle = math.degrees(-angle) % 360
            corrected_image = image.rotate(correction_angle, expand=True)

            # Save the corrected image to the output directory
            output_filename = os.path.join(
                output_dir, os.path.basename(img_path))
            corrected_image.save(output_filename)
            if v:
                print(
                    f"Processed image {i+1}/{len(image_paths)} named {img_path}")
                print(
                    f"Predicted correction angle: {correction_angle:.2f} degrees CW")

            predictions[os.path.basename(img_path)] = angle
        return predictions

    def predict_rotation(self, image: Image) -> float:
        '''
        Uses the model to predict the angle of rotation of the image.
        Args:
            image: the PIL image for prediction
        Returns:
            the angle of rotation of the image in radians
        '''
        # Predict the angle of rotation of the image
        if self._crop_strategy == "option1":
            # First Option
            batch = torch.stack([self._transforms(image.copy())
                                for _ in range(self._bs)]).to(self._device)
        elif self._crop_strategy == "option2":
            # Second Option
            batch = self._transforms2(
                image.copy()).unsqueeze(0).to(self._device)
        else:
            assert False, f"Unknown crop strategy [{self._crop_strategy}]"
        # TODO median? Evaluate which strategy is best?
        output = self._model(batch)
        angle = output.logits.detach().cpu().numpy().mean()
        # angle = np.median(output.logits.detach().cpu().numpy())
        return angle


def main_old(args):
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
            batch = torch.stack([_transforms(image.copy())
                                for _ in range(args.bs)]).to(device)
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
    parser.add_argument("--files", nargs="+", required=True,
                        help="Filenames of image files to process")
    parser.add_argument(
        "--output", help="Output directory to put rotated images")
    parser.add_argument("--bs", type=int, default=32,
                        help="Number of times for the model to look at each file")
    parser.add_argument("--crop_strategy", default='option1',
                        help="Method for cropping the image for the model")

    main(parser.parse_args())
