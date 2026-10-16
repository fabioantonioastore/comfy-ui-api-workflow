import functools
import json
import math
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import test_runpod_workflows as recording
from test_runpod_videos import (
    FFMPEG,
    TOOLS,
    Media,
    luminance,
    picture_difference,
    run_tool,
    spread,
    stream_seconds,
    wav,
)
from test_runpod_workflows import (
    GPU_TIMEOUT,
    png_size,
    run_and_record,
    sent,
    write_gallery,
    write_json,
)

from comfy_ui_api_workflow._impl.nodes.audio import AudioEqualizer3Band, LoadAudio
from comfy_ui_api_workflow._impl.nodes.image import LoadImage, SaveImage
from comfy_ui_api_workflow._impl.nodes.image.batch import ImageFromBatch
from comfy_ui_api_workflow._impl.nodes.model.conditioning import CLIPTextEncode
from comfy_ui_api_workflow._impl.nodes.model.conditioning.wan import (
    Wan22ImageToVideoLatent,
)
from comfy_ui_api_workflow._impl.nodes.model.latent import VAEDecode
from comfy_ui_api_workflow._impl.nodes.model.loaders import (
    CLIPLoader,
    UNETLoader,
    VAELoader,
)
from comfy_ui_api_workflow._impl.nodes.model.patch.stable_diffusion import (
    ModelSamplingSD3,
)
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler
from comfy_ui_api_workflow._impl.nodes.video import CreateVideo, SaveVideo
from comfy_ui_api_workflow._impl.pipeline import Pipeline
from comfy_ui_api_workflow._impl.workflow import Workflow

# A one-minute video of a lion running in the savanna, made with WAN 2.2 TI2V 5B on
# the endpoint's volume. WAN renders about 5 s (121 frames) per job, so the minute is
# four scenes of three clips; inside a scene each clip starts from the last frame of
# the one before. The clips are joined here with ffmpeg, without re-encoding them.
WAN_MODEL = "wan2.2_ti2v_5B_fp16.safetensors"
WAN_TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan2.2_vae.safetensors"
# WAN's own negative prompt, plus five legs, extra limbs, deformed animal, text, watermark.
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走，五条腿，多余的四肢，变形的动物，文字，水印"
)
FPS = 24
WIDTH, HEIGHT = 1280, 704
CLIP_FRAMES = 121
# The 121st frame starts the next clip, so each clip shows 120 frames: 5 s.
SHOWN = CLIP_FRAMES - 1
CLIP_SECONDS = SHOWN / FPS
PARTS = 3
MINUTE = 60
WIND_RATE = 22_050

LION = "a large adult male lion with a thick golden-brown mane"
DOCUMENTARY = "Cinematic wildlife documentary footage"
MOTION = "Realistic animal anatomy and natural running motion, sharp detail."
SCENES = (
    f"{DOCUMENTARY}. Wide tracking shot from a moving vehicle: {LION} sprints at full "
    "speed from left to right across the vast golden grass of the African savanna, its "
    "paws pounding the dry ground and kicking up dust, scattered acacia trees and "
    "distant hills behind it, warm early-morning sunlight, the camera smoothly follows "
    f"the lion. {MOTION}",
    f"{DOCUMENTARY}. Low-angle side tracking shot at ground level: {LION} gallops "
    "through tall dry savanna grass from left to right, shoulder muscles rippling, the "
    "mane bouncing with every stride, dust and grass seeds flying, golden afternoon "
    "light, shallow depth of field, the camera moves alongside the lion at the same "
    f"speed. {MOTION}",
    f"{DOCUMENTARY}. Frontal tracking shot: {LION} runs straight toward the camera "
    "across the open savanna, intense amber eyes, the mane flowing in the wind, front "
    "paws reaching forward with each bound, a dust cloud behind it, bright midday sun, "
    f"the camera moves backward to keep the lion in frame. {MOTION}",
    f"{DOCUMENTARY} at sunset. Side tracking shot: the silhouette of {LION} runs from "
    "left to right along a savanna ridge in front of a huge orange setting sun, acacia "
    "tree silhouettes, an orange and purple sky, glowing dust trailing behind it, the "
    f"camera follows the lion smoothly. {MOTION}",
)
SCENE_NAMES = (
    "manhã, plano aberto",
    "tarde, rente ao chão",
    "meio-dia, de frente",
    "pôr do sol",
)
FINAL_FOLDER = "000-especial-leao-na-savana"


# ---------------------------------------------------------------- the clips


class WanClip(Pipeline):
    """A WAN 2.2 TI2V 5B clip of 121 frames, from the prompt or from a start frame."""

    def build(self, prompt: str, seed: int, start_image: Any = None):
        unet = self.add_node(UNETLoader, unet_name=WAN_MODEL, weight_dtype="default")
        model = self.add_node(ModelSamplingSD3, model=unet.output["MODEL"], shift=8.0)
        text_encoder = self.add_node(
            CLIPLoader, clip_name=WAN_TEXT_ENCODER, type="wan", device="default"
        )
        clip = text_encoder.output["CLIP"]
        vae = self.add_node(VAELoader, vae_name=WAN_VAE).output["VAE"]
        start = {} if start_image is None else {"start_image": start_image}
        latent = self.add_node(
            Wan22ImageToVideoLatent,
            vae=vae,
            width=WIDTH,
            height=HEIGHT,
            length=CLIP_FRAMES,
            batch_size=1,
            **start,
        )
        sampled = self.add_node(
            KSampler,
            model=model.output["MODEL"],
            positive=self.add_node(CLIPTextEncode, text=prompt, clip=clip).output[
                "CONDITIONING"
            ],
            negative=self.add_node(CLIPTextEncode, text=WAN_NEGATIVE, clip=clip).output[
                "CONDITIONING"
            ],
            latent_image=latent.output["LATENT"],
            seed=seed,
            steps=20,
            cfg=5.0,
            sampler_name="uni_pc",
            scheduler="simple",
            denoise=1.0,
        )
        frames = self.add_node(VAEDecode, samples=sampled.output["LATENT"], vae=vae)
        return {"IMAGE": frames.output["IMAGE"]}


@dataclass(frozen=True)
class Clip:
    scene: int
    part: int

    @property
    def number(self) -> int:
        return self.scene * PARTS + self.part + 1

    @property
    def label(self) -> str:
        return f"leao_{self.number:02}"

    @property
    def following(self) -> str:
        return f"proximo_{self.number:02}"

    @property
    def folder(self) -> str:
        return f"000-especial-leao-parte-{self.number:02}"

    @property
    def start(self) -> str | None:
        return None if self.part == 0 else f"inicio_{self.number:02}.png"

    @property
    def sound(self) -> str:
        return f"vento_{self.number:02}.wav"

    @property
    def seed(self) -> int:
        return 2026 + 100 * self.scene + self.part

    @property
    def description(self) -> str:
        origin = (
            "do texto" if self.start is None else "do último quadro do trecho anterior"
        )
        return (
            f"Leão na savana, trecho {self.number} de {len(CLIPS)} (cena {self.scene + 1}: "
            f"{SCENE_NAMES[self.scene]}, parte {self.part + 1} de {PARTS}): WAN 2.2 TI2V 5B "
            f"em {WIDTH}x{HEIGHT}, {CLIP_FRAMES} quadros a {FPS} fps, partindo {origin}; "
            "mostra 120 quadros e devolve o 121º para o trecho seguinte, com o vento "
            "equalizado no ComfyUI."
        )


CLIPS = [Clip(scene, part) for scene in range(len(SCENES)) for part in range(PARTS)]


@functools.cache
def wind() -> tuple[float, ...]:
    # A minute of brown noise (integrated white noise) under two slow gusts.
    generator, level, samples = random.Random(7), 0.0, []
    for n in range(MINUTE * WIND_RATE):
        level = 0.996 * level + 0.06 * generator.uniform(-1.0, 1.0)
        t = n / WIND_RATE
        gust = 0.55 + 0.3 * math.sin(2 * math.pi * 0.11 * t)
        gust += 0.15 * math.sin(2 * math.pi * 0.037 * t + 1.3)
        samples.append(level * gust)
    peak = max(abs(value) for value in samples)
    return tuple(0.5 * value / peak for value in samples)


def wind_of(clip: Clip) -> bytes:
    size = round(CLIP_SECONDS * WIND_RATE)
    start = (clip.number - 1) * size
    return wav(WIND_RATE, list(wind()[start : start + size]))


def clip_workflow(clip: Clip) -> Workflow:
    w = Workflow()
    start = {}
    if clip.start is not None:
        start["start_image"] = w.add_node(LoadImage, image=clip.start).output["IMAGE"]
    frames = w.add_pipeline(
        WanClip, prompt=SCENES[clip.scene], seed=clip.seed, **start
    ).output["IMAGE"]
    shown = w.add_node(ImageFromBatch, image=frames, batch_index=0, length=SHOWN)
    following = w.add_node(ImageFromBatch, image=frames, batch_index=SHOWN, length=1)
    sound = w.add_node(
        AudioEqualizer3Band,
        audio=w.add_node(LoadAudio, audio=clip.sound).output["AUDIO"],
        low_gain_dB=3.0,
        low_freq=200,
        mid_gain_dB=-6.0,
        mid_freq=1000,
        mid_q=0.7,
        high_gain_dB=-18.0,
        high_freq=3000,
    ).output["AUDIO"]
    video = w.add_node(
        CreateVideo, images=shown.output["IMAGE"], fps=float(FPS), audio=sound
    ).output["VIDEO"]
    w.add_node(
        SaveVideo,
        video=video,
        filename_prefix=clip.label,
        format="mp4",
        format_codec="h264",
    )
    w.add_node(
        SaveImage, images=following.output["IMAGE"], filename_prefix=clip.following
    )
    return w


def uploads_of(clip: Clip, start_png: bytes | None) -> dict[str, bytes]:
    uploads = {clip.sound: wind_of(clip)}
    if clip.start is not None:
        assert start_png is not None, f"clip {clip.number} needs the frame before it"
        uploads[clip.start] = start_png
    return uploads


def check_clip(clip: Clip):
    def check(out):
        media = Media.returned(out, clip.label)
        assert media.size == (WIDTH, HEIGHT), f"the clip is {media.size}"
        assert media.fps == FPS, f"the clip runs at {media.fps} fps"
        assert media.frames == SHOWN, f"the clip has {media.frames} frames"
        assert (
            abs(media.seconds - CLIP_SECONDS) <= 0.1
        ), f"the clip lasts {media.seconds}s"
        assert (media.audio["codec_name"], media.audio["channels"]) == ("aac", 1)
        assert int(media.audio["sample_rate"]) == WIND_RATE
        assert abs(stream_seconds(media.audio) - CLIP_SECONDS) <= 0.1
        assert png_size(out.one(clip.following)) == (WIDTH, HEIGHT)
        first, last = media.frame(0.5), media.frame(4.5)
        assert spread(luminance(first)) > 15, "the clip is flat"
        assert picture_difference(first, last) > 2, "nothing moves in the clip"

    return check


# ---------------------------------------------------------------- rendering


def only_file(folder: Path, pattern: str) -> Path:
    found = sorted(folder.glob(pattern))
    assert len(found) == 1, f"expected one {pattern} in {folder.name}, got {found}"
    return found[0]


def already_rendered(folder: Path, payload: dict, uploads: dict[str, bytes]) -> bool:
    # A clip is only reused when the same job, with the same files, passed before.
    try:
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        saved = json.loads((folder / "workflow.json").read_text(encoding="utf-8"))
        sent_files = {
            name: (folder / f"entrada-{name}").read_bytes() for name in uploads
        }
    except (OSError, ValueError):
        return False
    return (
        result.get("status") == "passed"
        and saved == payload
        and sent_files == uploads
        and len(list(folder.glob("leao_*.mp4"))) == 1
        and len(list(folder.glob("proximo_*.png"))) == 1
    )


def render_scene(endpoint, scene: int) -> list[Path]:
    videos, start_png = [], None
    for part in range(PARTS):
        clip = Clip(scene, part)
        payload = sent(clip_workflow(clip))
        uploads = uploads_of(clip, start_png)
        folder = recording.GALLERY / clip.folder
        if not already_rendered(folder, payload, uploads):
            run_and_record(
                endpoint,
                clip.folder,
                f"leao-{clip.number:02}",
                clip.description,
                payload,
                timeout=GPU_TIMEOUT,
                uploads=uploads,
                check=check_clip(clip),
            )
        videos.append(only_file(folder, f"{clip.label}_*.mp4"))
        start_png = only_file(folder, f"{clip.following}_*.png").read_bytes()
    return videos


def assemble(videos: list[Path], target: Path) -> None:
    # The clips share every encoder setting, so the video is copied as is; the sound
    # is decoded and encoded again, which joins it without gaps.
    listing = target.with_name(f"{target.stem}-lista.txt")
    listing.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in videos), encoding="utf-8"
    )
    try:
        run_tool(
            FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(target),
        )
    finally:
        listing.unlink()


def contact_sheet(video: Path, target: Path, frames: list[int]) -> None:
    chosen = "+".join(f"eq(n\\,{frame})" for frame in frames)
    run_tool(
        FFMPEG,
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select='{chosen}',scale=480:-2,tile={len(frames)}x1",
        "-frames:v",
        "1",
        str(target),
    )


def scene_boundaries() -> list[float]:
    # Where one clip goes on from the one before, inside a scene (not the cuts).
    return [clip.number - 1 for clip in CLIPS if clip.part > 0]


def check_minute(media: Media) -> None:
    assert abs(media.seconds - MINUTE) <= 0.1, f"the video lasts {media.seconds}s"
    assert media.size == (WIDTH, HEIGHT), f"the video is {media.size}"
    assert media.fps == FPS, f"the video runs at {media.fps} fps"
    assert media.frames == MINUTE * FPS, f"the video has {media.frames} frames"
    assert (media.audio["codec_name"], media.audio["channels"]) == ("aac", 1)
    seconds = stream_seconds(media.audio)
    assert abs(seconds - MINUTE) <= 0.15, f"the wind lasts {seconds}s"
    sound = media.sound()
    for second in range(MINUTE):
        assert sound.rms(second, second + 1) > 0.005, f"the wind stops at {second}s"
    step = 1 / FPS
    for index in scene_boundaries():
        at = index * CLIP_SECONDS
        motion = [
            picture_difference(
                media.frame(at - (back + 2) * step), media.frame(at - (back + 1) * step)
            )
            for back in range(3)
        ]
        across = picture_difference(media.frame(at - step), media.frame(at))
        assert (
            across <= 2.5 * max(motion) + 6
        ), f"the clips jump at {at}s: {across:.1f} against a motion of {max(motion):.1f}"


# ---------------------------------------------------------------- tests


def nodes_of(payload: dict, class_type: str) -> list[dict]:
    return [
        node["inputs"] for node in payload.values() if node["class_type"] == class_type
    ]


def test_lion_clip_follows_the_wan_2_2_5b_template():
    payload = sent(clip_workflow(CLIPS[0]))

    assert nodes_of(payload, "UNETLoader") == [
        {"unet_name": WAN_MODEL, "weight_dtype": "default"}
    ]
    ((text_encoder),) = nodes_of(payload, "CLIPLoader")
    assert (text_encoder["clip_name"], text_encoder["type"]) == (
        WAN_TEXT_ENCODER,
        "wan",
    )
    assert nodes_of(payload, "ModelSamplingSD3")[0]["shift"] == 8.0
    ((sampler),) = nodes_of(payload, "KSampler")
    assert (
        sampler["steps"],
        sampler["cfg"],
        sampler["sampler_name"],
        sampler["scheduler"],
    ) == (
        20,
        5.0,
        "uni_pc",
        "simple",
    )
    ((latent),) = nodes_of(payload, "Wan22ImageToVideoLatent")
    assert (latent["width"], latent["height"], latent["length"]) == (
        WIDTH,
        HEIGHT,
        CLIP_FRAMES,
    )
    assert "start_image" not in latent
    assert nodes_of(payload, "CreateVideo")[0]["fps"] == FPS
    batches = [
        (node["batch_index"], node["length"])
        for node in nodes_of(payload, "ImageFromBatch")
    ]
    assert sorted(batches) == [(0, SHOWN), (SHOWN, 1)]


def test_lion_clip_goes_on_from_the_frame_it_is_sent():
    clip = CLIPS[1]
    payload = sent(clip_workflow(clip))
    ((loader_id, _),) = [
        (node_id, node)
        for node_id, node in payload.items()
        if node["class_type"] == "LoadImage"
    ]

    ((latent),) = nodes_of(payload, "Wan22ImageToVideoLatent")
    assert payload[loader_id]["inputs"]["image"] == clip.start == "inicio_02.png"
    assert latent["start_image"] == [loader_id, 0]
    assert set(uploads_of(clip, b"png")) == {"inicio_02.png", "vento_02.wav"}
    with pytest.raises(AssertionError, match="needs the frame before it"):
        uploads_of(clip, None)


def test_lion_minute_is_twelve_clips_of_five_seconds():
    assert len(CLIPS) == len(SCENES) * PARTS == 12
    assert len(CLIPS) * SHOWN == MINUTE * FPS
    assert CLIP_FRAMES == SHOWN + 1 and (CLIP_FRAMES - 1) % 4 == 0
    assert [clip.start is None for clip in CLIPS] == [True, False, False] * len(SCENES)
    assert len({clip.seed for clip in CLIPS}) == len(CLIPS)
    assert scene_boundaries() == [1, 2, 4, 5, 7, 8, 10, 11]


def test_lion_wind_is_one_minute_cut_into_the_clips():
    chunks = [wind_of(clip) for clip in CLIPS]
    whole = wav(WIND_RATE, list(wind()))
    header = 44

    assert len(wind()) == MINUTE * WIND_RATE
    assert b"".join(chunk[header:] for chunk in chunks) == whole[header:]
    assert 0.05 < math.sqrt(sum(value * value for value in wind()) / len(wind())) < 0.3


@pytest.mark.parametrize("clip", CLIPS, ids=[clip.label for clip in CLIPS])
def test_lion_clip_workflow_survives_a_reload(clip):
    payload = sent(clip_workflow(clip))
    workflow = Workflow()

    workflow.load_workflow(json.loads(json.dumps(payload)))

    assert sent(workflow) == payload


def test_a_clip_is_reused_only_when_the_same_job_passed(tmp_path):
    payload, uploads = {"1": {"class_type": "EmptyImage", "inputs": {}}}, {
        "vento_01.wav": b"wind"
    }
    write_json(tmp_path / "result.json", {"status": "passed"})
    write_json(tmp_path / "workflow.json", payload)
    (tmp_path / "entrada-vento_01.wav").write_bytes(b"wind")
    (tmp_path / "leao_01_00001_.mp4").write_bytes(b"")
    (tmp_path / "proximo_01_00001_.png").write_bytes(b"")

    assert already_rendered(tmp_path, payload, uploads)
    assert not already_rendered(tmp_path, {"2": {}}, uploads)
    assert not already_rendered(tmp_path, payload, {"vento_01.wav": b"other"})
    write_json(tmp_path / "result.json", {"status": "failed"})
    assert not already_rendered(tmp_path, payload, uploads)


@pytest.mark.skipif(not TOOLS, reason="needs ffmpeg and ffprobe")
def test_assemble_joins_the_clips_in_order(tmp_path):
    colors = ("red", "lime", "blue")
    clips = []
    for index, color in enumerate(colors):
        path = tmp_path / f"clip_{index}.mp4"
        subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=64x48:r=24:d=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=300:sample_rate=22050:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ac",
                "1",
                str(path),
            ],
            check=True,
        )
        clips.append(path)
    target = tmp_path / "joined.mp4"

    assemble(clips, target)

    media = Media(target.read_bytes(), ".mp4")
    assert media.frames == 72 and abs(media.seconds - 3.0) < 0.1
    assert [media.frame(index + 0.5).pixel(32, 24)[1] > 200 for index in range(3)] == [
        False,
        True,
        False,
    ]
    assert (
        media.frame(0.5).pixel(32, 24)[0] > 200
        and media.frame(2.5).pixel(32, 24)[2] > 200
    )
    sound = media.sound()
    assert all(sound.rms(second + 0.1, second + 0.9) > 0.05 for second in range(3))
    assert not list(tmp_path.glob("*-lista.txt"))


@pytest.mark.runpod
def test_lion_runs_for_a_minute_in_the_savanna(runpod_endpoint):
    if not TOOLS:
        pytest.skip("ffmpeg and ffprobe are needed to join and measure the clips")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(SCENES)) as pool:
        scenes = list(
            pool.map(
                lambda scene: render_scene(runpod_endpoint, scene), range(len(SCENES))
            )
        )
    videos = [video for scene in scenes for video in scene]
    folder = recording.GALLERY / FINAL_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    parts = [
        json.loads((video.parent / "result.json").read_text(encoding="utf-8"))
        for video in videos
    ]
    result: dict[str, Any] = {
        "title": "leao-na-savana",
        "description": (
            "Especial: um leão correndo na savana por um minuto. Doze trechos de 5 s do WAN "
            "2.2 TI2V 5B em quatro cenas (manhã, tarde rente ao chão, meio-dia de frente e "
            "pôr do sol); dentro de cada cena um trecho continua o anterior a partir do "
            "último quadro dele. Os trechos foram gerados no RunPod e emendados com ffmpeg."
        ),
        "status": "running",
        "nodes": sum(part["nodes"] for part in parts),
        "class_types": sorted({kind for part in parts for kind in part["class_types"]}),
        "execution_ms": sum(part.get("execution_ms") or 0 for part in parts),
        "queue_ms": sum(part.get("queue_ms") or 0 for part in parts),
    }
    write_json(
        folder / "workflow.json", {"trechos": [video.parent.name for video in videos]}
    )
    target = folder / "leao_na_savana_1_minuto.mp4"
    try:
        assemble(videos, target)
        contact_sheet(
            target,
            folder / "cenas.png",
            [180 + 360 * scene for scene in range(len(SCENES))],
        )
        check_minute(Media(target.read_bytes(), ".mp4"))
    except BaseException as error:
        result.update(status="failed", message=str(error)[:6000])
        raise
    else:
        result["status"] = "passed"
    finally:
        result["seconds"] = round(time.monotonic() - started, 1)
        write_json(folder / "result.json", result)
        write_gallery()
