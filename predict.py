from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image
import torch
from controlnet_aux.pidi import PidiNetDetector
import shutil
from PIL import Image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionXLAdapterPipeline.from_single_file("juggernaut.safetensors", adapter=adapter, torch_dtype=torch.float16, use_safetensors=True)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        self.pipe.to("cuda")
        self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")

    def resize_square(self, image: Image.Image, size: int, fill_color=(0, 0, 0)):
        """
        Resize and pad an image to make it square by resizing the longer side.

        Args:
            image (PIL.Image.Image): The input image.
            size (int): The desired size of the square image (width and height).
            fill_color (tuple, optional): The color to use for padding. Defaults to white (255, 255, 255).

        Returns:
            PIL.Image.Image: The square image.
        """
        # Get the dimensions of the input image
        width, height = image.size

        # Determine the longer side and calculate the scaling factor
        if width > height:
            scale_factor = size / width
        else:
            scale_factor = size / height

        # Calculate the new dimensions after resizing
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image using the calculated dimensions
        resized_image = image.resize((new_width, new_height), Image.BICUBIC)

        # Create a new image with the desired size and fill it with the specified color
        square_image = Image.new("RGB", (size, size), fill_color)

        # Calculate the position to paste the resized image in the center of the square
        x_offset = (size - new_width) // 2
        y_offset = (size - new_height) // 2

        # Paste the resized image onto the square canvas
        square_image.paste(resized_image, (x_offset, y_offset))

        return square_image

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
            default="textured, high quality, full detailed material, studio style, simple background, dslr, natural lighting, shot by camera, RAW image, photorealistic, sharp focus, 8k, uhd, file grain, masterpiece"
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="deformed, animation, anime, cartoon, comic, cropped, out of frame, low res, draft, cgi, low quality render, thumbnail",
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
    ) -> Path:
        """Run a single prediction on the model"""
        conditional_image = self.load_image(image)
        conditional_image = self.pidinet(conditional_image, detect_resolution=1024, image_resolution=1024, apply_filter=True)
        if generate_square:
            conditional_image = self.resize_square(conditional_image, 1024)
            height = 1024
            width = 1024
        if seed:
            generator = torch.manual_seed(seed)
        else:
            generator = None
        prompt = f"{prompt}, {suffix_prompt}"

        generated_images = self.pipe(
            prompt,
            negative_prompt = negative_prompt,
            image=conditional_image,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_outputs,
            adapter_conditioning_scale = adapter_conditioning_scale,
        ).images

        output_paths = []
        for i, image in enumerate(generated_images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
