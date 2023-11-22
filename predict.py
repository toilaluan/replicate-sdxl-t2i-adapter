from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
)
from diffusers.utils import load_image
import torch
from controlnet_aux.pidi import PidiNetDetector
from controlnet_aux.canny import CannyDetector
import shutil
from typing import List
from PIL import Image
import numpy as np


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        adapter = T2IAdapter.from_pretrained(
            "/controlnet", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionXLAdapterPipeline.from_single_file(
            "/juggernaut.safetensors",
            adapter=adapter,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.to("cuda:2")
        self.pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        self.canny_detector = CannyDetector()

    from PIL import Image

    def resize_and_pad(self, img, target_w, target_h, fill_color=(0, 0, 0)):
        """
        Resize a PIL image to target width and height, padding with a fill color.

        :param img: PIL Image object.
        :param target_w: Target width.
        :param target_h: Target height.
        :param fill_color: Fill color for padding (R, G, B).
        :return: Resized and padded image.
        """
        # Calculate the aspect ratio of the input image and the target size.
        original_w, original_h = img.size
        ratio_img = original_w / original_h
        ratio_target = target_w / target_h

        # Resize the image based on the aspect ratio.
        if ratio_img > ratio_target:
            # Image is wider than target aspect ratio.
            new_height = int(original_h * target_w / original_w)
            resized_img = img.resize((target_w, new_height), Image.LANCZOS)
        else:
            # Image is taller than target aspect ratio.
            new_width = int(original_w * target_h / original_h)
            resized_img = img.resize((new_width, target_h), Image.LANCZOS)

        # Create a new image with the specified fill color and target size.
        new_img = Image.new("RGB", (target_w, target_h), fill_color)

        # Calculate the position to paste the resized image onto the new image.
        paste_x = (target_w - resized_img.size[0]) // 2
        paste_y = (target_h - resized_img.size[1]) // 2

        # Paste the resized image onto the new image.
        new_img.paste(resized_img, (paste_x, paste_y))

        return new_img

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a running shoe",
        ),
        suffix_prompt: str = Input(
            description="Additional prompt",
            default="textured, high quality, full detailed material, studio style, simple background, dslr, natural lighting, shot by camera, RAW image, photorealistic, sharp focus, 8k, uhd, file grain, masterpiece",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="deformed, animation, anime, cartoon, comic, cropped, out of frame, low res, draft, cgi, low quality render, thumbnail",
        ),
        use_canny: bool = Input(
            description="Whether to use canny detector for better details",
            default=False,
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        generate_square: bool = Input(
            description="Whether generate square image, assert height == width",
            default=False,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=35
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        adapter_conditioning_scale: float = Input(
            description="Scale for adapter module", ge=0, le=2, default=0.9
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        rgb_conditional_image = self.load_image(image)
        sketch_conditional_image = self.pidinet(
            rgb_conditional_image,
            detect_resolution=1024,
            image_resolution=1024,
            apply_filter=True,
        )
        canny_conditional_image = self.canny_detector(
            rgb_conditional_image, detect_resolution=1024, image_resolution=1024
        )
        if use_canny:
            conditional_image = Image.fromarray(
                np.array(canny_conditional_image) + np.array(sketch_conditional_image)
            )
        else:
            conditional_image = sketch_conditional_image

        if generate_square:
            conditional_image = self.resize_and_pad(conditional_image, 1024, 1024)
            height = 1024
            width = 1024
        else:
            conditional_image = self.resize_and_pad(conditional_image, width, height)
        if seed:
            generator = torch.manual_seed(seed)
        else:
            generator = None
        prompt = f"{prompt}, {suffix_prompt}"

        generated_images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=conditional_image,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_outputs,
            adapter_conditioning_scale=adapter_conditioning_scale,
        ).images

        output_paths = []
        for i, image in enumerate(generated_images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))
        conditional_image.save(f"/tmp/out-{i+1}.png")
        output_paths.append(Path(f"/tmp/out-{i+1}.png"))

        return output_paths
