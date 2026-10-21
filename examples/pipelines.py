from comfy_ui_api_workflow._impl.nodes.image import SaveImage
from comfy_ui_api_workflow._impl.nodes.image.upscaling import ImageScaleBy
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.latent import EmptyLatentImage, VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.loaders import CheckpointLoaderSimple
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.nodes.video import CreateVideo, SaveVideo
from comfy_ui_api_workflow._impl.pipeline import Pipeline

Link = tuple[str, int]


class TextToImage(Pipeline):
    def build(  # type: ignore
        self,
        prompt: str,
        negative: str = "blurry, low quality, distorted",
        checkpoint: str = "sd_xl_base_1.0.safetensors",
        size: int = 1024,
        steps: int = 25,
        seed: int = 42,
    ):
        model = self.add_node(CheckpointLoaderSimple, ckpt_name=checkpoint)
        positive_text = self.add_node(
            CLIPTextEncode, text=prompt, clip=model.output["CLIP"]
        )
        negative_text = self.add_node(
            CLIPTextEncode, text=negative, clip=model.output["CLIP"]
        )
        latent = self.add_node(EmptyLatentImage, width=size, height=size, batch_size=1)
        sampler = self.add_node(
            KSampler,
            model=model.output["MODEL"],
            sampler_name="dpmpp_2m",
            scheduler="karras",
            positive=positive_text.output["CONDITIONING"],
            negative=negative_text.output["CONDITIONING"],
            latent_image=latent.output["LATENT"],
            seed=seed,
            steps=steps,
            cfg=7.0,
        )
        image = self.add_node(
            VAEDecode, samples=sampler.output["LATENT"], vae=model.output["VAE"]
        )
        return {"IMAGE": image.output["IMAGE"]}


class Upscale(Pipeline):
    def build(self, image: Link, scale: float = 2.0):  # type: ignore
        scaled = self.add_node(
            ImageScaleBy, image=image, upscale_method="lanczos", scale_by=scale
        )
        return {"IMAGE": scaled.output["IMAGE"]}


class SaveImages(Pipeline):
    def build(self, images: Link, prefix: str = "ComfyUI"):  # type: ignore
        self.add_node(SaveImage, images=images, filename_prefix=prefix)


class ImagesToVideo(Pipeline):
    def build(self, frames: Link, fps: float = 8.0, prefix: str = "video/ComfyUI"):  # type: ignore
        video = self.add_node(CreateVideo, images=frames, fps=fps)
        self.add_node(
            SaveVideo,
            video=video.output["VIDEO"],
            filename_prefix=prefix,
            format="mp4",
            format_codec="h264",
        )
        return {"VIDEO": video.output["VIDEO"]}
