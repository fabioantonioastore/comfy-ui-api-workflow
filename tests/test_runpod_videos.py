import io
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import pytest
from test_runpod_workflows import (
    GPU_TIMEOUT,
    NEGATIVE,
    TIMEOUT,
    Outputs,
    Picture,
    assert_pixel,
    decode,
    empty_latent,
    hues,
    ksample,
    media_html,
    prompt,
    rgb,
    run_and_record,
    scale,
    sdxl,
    sent,
    solid,
    stitch,
)

from comfy_ui_api_workflow._impl.nodes.audio import (
    AudioAdjustVolume,
    AudioConcat,
    AudioEqualizer3Band,
    AudioMerge,
    EmptyAudio,
    JoinAudioChannels,
    LoadAudio,
    SplitAudioChannels,
    TrimAudioDuration,
)
from comfy_ui_api_workflow._impl.nodes.image.batch import (
    RebatchImages,
    RepeatImageBatch,
    SplitImageToTileList,
)
from comfy_ui_api_workflow._impl.nodes.image.color import ImageInvert
from comfy_ui_api_workflow._impl.nodes.image.filters import ImageAddNoise, ImageBlend
from comfy_ui_api_workflow._impl.nodes.video import (
    CreateVideo,
    GetVideoComponents,
    SaveVideo,
    VideoSlice,
)
from comfy_ui_api_workflow._impl.workflow import Workflow

# Ten 30-second videos with sound, each built and checked differently. The sound
# starts as short WAV files made here and sent with the job; the workflow builds
# the soundtrack from them. The returned files are measured with ffprobe/ffmpeg.
SECONDS = 30
ANALYSIS_RATE = 16_000
FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
TOOLS = FFMPEG is not None and FFPROBE is not None

WHITE, BLACK, NIGHT = 0xFFFFFF, 0x000000, 0x101828
YELLOW, NAVY = 0xFFD000, 0x14205A
RED, GREEN, BLUE = 0xE02020, 0x20B040, 0x2050E0


# ---------------------------------------------------------------- sound files


def wav(rate: int, *channels: list[float]) -> bytes:
    samples = array(
        "h",
        (
            round(max(-1.0, min(1.0, value)) * 32767)
            for frame in zip(*channels)
            for value in frame
        ),
    )
    if sys.byteorder == "big":
        samples.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(len(channels))
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.tobytes())
    return buffer.getvalue()


def sine(frequency: float, seconds: float, rate: int, amplitude: float) -> list[float]:
    step = 2 * math.pi * frequency / rate
    return [amplitude * math.sin(step * n) for n in range(round(seconds * rate))]


def chord(frequencies, seconds: float, rate: int, amplitude: float) -> list[float]:
    tones = [sine(frequency, seconds, rate, amplitude) for frequency in frequencies]
    return [sum(values) for values in zip(*tones)]


def hiss(seconds: float, rate: int, amplitude: float, seed: int) -> list[float]:
    generator = random.Random(seed)
    return [
        amplitude * generator.uniform(-1.0, 1.0) for _ in range(round(seconds * rate))
    ]


def quiet(seconds: float, rate: int) -> list[float]:
    return [0.0] * round(seconds * rate)


def faded(samples: list[float], rate: int, seconds: float = 0.005) -> list[float]:
    # A short ramp at both ends, so the click starts and stops without a pop.
    ramp = round(seconds * rate)
    return [
        value * min(1.0, (index + 1) / ramp, (len(samples) - index) / ramp)
        for index, value in enumerate(samples)
    ]


# ---------------------------------------------------------------- measuring


@dataclass(frozen=True)
class Sound:
    rate: int
    channels: list[list[float]]

    def part(self, start: float, end: float, channel: int = 0) -> list[float]:
        return self.channels[channel][round(start * self.rate) : round(end * self.rate)]

    def rms(self, start: float, end: float, channel: int = 0) -> float:
        samples = self.part(start, end, channel)
        return math.sqrt(sum(value * value for value in samples) / len(samples))

    def amplitude(
        self, frequency: float, start: float, end: float, channel: int = 0
    ) -> float:
        # Goertzel: the amplitude of the sine at this exact frequency.
        samples = self.part(start, end, channel)
        coefficient = 2 * math.cos(2 * math.pi * frequency / self.rate)
        previous = before = 0.0
        for value in samples:
            previous, before = value + coefficient * previous - before, previous
        power = previous * previous + before * before - coefficient * previous * before
        return 2 * math.sqrt(max(power, 0.0)) / len(samples)

    def dominant(
        self, frequencies, start: float, end: float, channel: int = 0
    ) -> float:
        return max(
            frequencies,
            key=lambda frequency: self.amplitude(frequency, start, end, channel),
        )

    def band(self, frequencies, start: float, end: float, window: float = 0.2) -> float:
        # The mean level of noise around these frequencies, over many short windows.
        powers = [
            self.amplitude(frequency, at, at + window) ** 2
            for at in frange(start, end - window, window)
            for frequency in frequencies
        ]
        return math.sqrt(sum(powers) / len(powers))

    def envelope(
        self, start: float, end: float, window: float, channel: int = 0
    ) -> list[float]:
        return [
            self.rms(at, at + window, channel)
            for at in frange(start, end - window, window)
        ]


def frange(start: float, stop: float, step: float) -> list[float]:
    return [
        start + index * step for index in range(int((stop - start) / step + 1e-9) + 1)
    ]


def decibels(value: float) -> float:
    return 20 * math.log10(value)


def stream_seconds(stream: dict) -> float:
    if "duration" in stream:
        return float(stream["duration"])
    # Matroska keeps each stream's length in a DURATION tag, as HH:MM:SS.fraction.
    hours, minutes, seconds = stream["tags"]["DURATION"].split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def run_tool(*args: str) -> bytes:
    return subprocess.run(args, capture_output=True, check=True).stdout


class Media:
    """A video file, measured with ffprobe and decoded with ffmpeg."""

    def __init__(self, data: bytes, suffix: str) -> None:
        self.__folder = tempfile.TemporaryDirectory(prefix="video-")
        self.path = Path(self.__folder.name) / f"media{suffix}"
        self.path.write_bytes(data)
        info = json.loads(
            run_tool(
                FFPROBE,
                "-v",
                "error",
                "-count_frames",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(self.path),
            )
        )
        self.format = info["format"]
        streams = info["streams"]
        (self.video,) = [
            stream for stream in streams if stream["codec_type"] == "video"
        ]
        audio = [stream for stream in streams if stream["codec_type"] == "audio"]
        assert len(audio) == 1, f"the video has {len(audio)} audio tracks"
        self.audio = audio[0]

    @classmethod
    def returned(cls, out: Outputs, label: str) -> "Media":
        return cls(out.one(label), Path(out.name(label)).suffix)

    @property
    def seconds(self) -> float:
        return float(self.format["duration"])

    @property
    def size(self) -> tuple[int, int]:
        return self.video["width"], self.video["height"]

    @property
    def fps(self) -> Fraction:
        return Fraction(self.video["avg_frame_rate"])

    @property
    def frames(self) -> int:
        return int(self.video["nb_read_frames"])

    def sound(self, channels: int = 1) -> Sound:
        raw = run_tool(
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(self.path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(channels),
            "-ar",
            str(ANALYSIS_RATE),
            "-",
        )
        samples = array("h")
        samples.frombytes(raw)
        if sys.byteorder == "big":
            samples.byteswap()
        return Sound(
            ANALYSIS_RATE,
            [
                [value / 32768 for value in samples[channel::channels]]
                for channel in range(channels)
            ],
        )

    def frame(self, seconds: float) -> Picture:
        # ffmpeg's -ss gives the first frame at or after that time, so the seek goes
        # just before the start of the frame shown at `seconds`.
        fps = float(self.fps)
        shown = math.floor(seconds * fps + 1e-6)
        raw = run_tool(
            FFMPEG,
            "-v",
            "error",
            "-ss",
            f"{max(0.0, (shown - 0.25) / fps):.4f}",
            "-i",
            str(self.path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        )
        width, height = self.size
        assert (
            len(raw) == width * height * 3
        ), f"frame at {seconds}s has {len(raw)} bytes"
        line = width * 3
        return Picture(
            width,
            height,
            3,
            tuple(raw[y * line : (y + 1) * line] for y in range(height)),
        )


def assert_thirty_seconds(
    media: Media,
    *,
    fps: float,
    size: tuple[int, int],
    video: str,
    audio: str,
    rate: int,
    channels: int,
    frame_slack: int = 0,
):
    # One frame of slack also lets the length run one frame over.
    slack = 0.1 + frame_slack / fps
    assert abs(media.seconds - SECONDS) <= slack, f"the file lasts {media.seconds}s"
    assert media.video["codec_name"] == video, media.video["codec_name"]
    assert media.size == size, f"the video is {media.size}, expected {size}"
    # Matroska stores times in milliseconds, so its average rate can be a near fraction.
    assert abs(media.fps - Fraction(fps)) < 0.01, f"the video runs at {media.fps} fps"
    assert (
        abs(media.frames - fps * SECONDS) <= frame_slack
    ), f"the video has {media.frames} frames"
    assert abs(stream_seconds(media.video) - SECONDS) <= slack
    assert media.audio["codec_name"] == audio, media.audio["codec_name"]
    assert int(media.audio["sample_rate"]) == rate, media.audio["sample_rate"]
    assert media.audio["channels"] == channels, media.audio["channels"]
    seconds = stream_seconds(media.audio)
    assert abs(seconds - SECONDS) <= 0.1, f"the sound lasts {seconds}s"


def luminance(picture: Picture, step: int = 4) -> list[float]:
    return [
        0.299 * r + 0.587 * g + 0.114 * b
        for y in range(0, picture.height, step)
        for x in range(0, picture.width, step)
        for r, g, b in [picture.pixel(x, y)]
    ]


def spread(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def picture_difference(first: Picture, second: Picture) -> float:
    return sum(abs(a - b) for a, b in zip(luminance(first), luminance(second))) / len(
        luminance(first)
    )


def inverted(color: int) -> int:
    return color ^ 0xFFFFFF


# ---------------------------------------------------------------- the registry


@dataclass(frozen=True)
class Video:
    name: str
    description: str
    build: Callable[[Workflow], Callable[[Outputs], None]]
    fps: float
    uploads: dict[str, bytes] = field(default_factory=dict)
    gpu: bool = False


VIDEOS: list[Video] = []


def video(
    description: str, *, fps: float, uploads: dict[str, bytes], gpu: bool = False
):
    def register(build: Callable) -> Callable:
        VIDEOS.append(Video(build.__name__, description, build, fps, uploads, gpu))
        return build

    return register


# ---------------------------------------------------------------- building blocks


def load(w: Workflow, name: str):
    return w.add_node(LoadAudio, audio=name).output["AUDIO"]


def concat(w: Workflow, first, second, direction: str = "after"):
    return w.add_node(
        AudioConcat, audio1=first, audio2=second, direction=direction
    ).output["AUDIO"]


def doubled(w: Workflow, audio, times: int):
    for _ in range(times):
        audio = concat(w, audio, audio)
    return audio


def trim(w: Workflow, audio, start: float, duration: float):
    return w.add_node(
        TrimAudioDuration, audio=audio, start_index=start, duration=duration
    ).output["AUDIO"]


def louder(w: Workflow, audio, decibels: int):
    return w.add_node(AudioAdjustVolume, audio=audio, volume=decibels).output["AUDIO"]


def merge(w: Workflow, first, second, method: str):
    return w.add_node(
        AudioMerge, audio1=first, audio2=second, merge_method=method
    ).output["AUDIO"]


def stereo(w: Workflow, left, right):
    return w.add_node(JoinAudioChannels, audio_left=left, audio_right=right).output[
        "audio"
    ]


def left_channel(w: Workflow, audio):
    return w.add_node(SplitAudioChannels, audio=audio).output["left"]


def pause(w: Workflow, seconds: float, rate: int):
    return w.add_node(
        EmptyAudio, duration=seconds, sample_rate=rate, channels=1
    ).output["AUDIO"]


def side_by_side(w: Workflow, images):
    strip = images[0]
    for image in images[1:]:
        strip = stitch(w, strip, image)
    return strip


def color_strip(w: Workflow, colors, size: tuple[int, int]):
    return side_by_side(w, [solid(w, size[0], size[1], color) for color in colors])


def timeline(w: Workflow, strip, size: tuple[int, int], total: int, hold: int = 1):
    # The strip is cut into frame-sized tiles, a list in left-to-right order; each
    # tile (a frame or a whole batch) is held for `hold` frames and the list is
    # rebatched into one batch of `total` frames.
    tiles = w.add_node(
        SplitImageToTileList,
        image=strip,
        tile_width=size[0],
        tile_height=size[1],
        overlap=0,
    ).output["IMAGE"]
    if hold > 1:
        tiles = w.add_node(RepeatImageBatch, image=tiles, amount=hold).output["IMAGE"]
    return w.add_node(RebatchImages, images=tiles, batch_size=total).output["IMAGE"]


def repeated(w: Workflow, images, amount: int):
    return w.add_node(RepeatImageBatch, image=images, amount=amount).output["IMAGE"]


def dimmed(w: Workflow, image, size: tuple[int, int], darkness: float):
    black = solid(w, size[0], size[1], BLACK)
    return w.add_node(
        ImageBlend,
        image1=image,
        image2=black,
        blend_factor=darkness,
        blend_mode="normal",
    ).output["IMAGE"]


def movie(
    w: Workflow, images, fps: float, audio, label: str, container="mp4", codec="h264"
):
    clip = w.add_node(CreateVideo, images=images, fps=fps, audio=audio).output["VIDEO"]
    w.add_node(
        SaveVideo,
        video=clip,
        filename_prefix=label,
        format=container,
        format_codec=codec,
    )


# ---------------------------------------------------------------- 1. metronome

CLICK_RATE = 22_050


@video(
    "Metrônomo: um clique de 0,1 s a 1 kHz (WAV enviado) é dobrado cinco vezes com "
    "AudioConcat, que o deixa estéreo, e cortado em 30 s com TrimAudioDuration; a tela "
    "pisca branca no primeiro quarto de cada segundo, junto com a batida.",
    fps=4.0,
    uploads={
        "clique.wav": wav(
            CLICK_RATE,
            faded(sine(1000, 0.1, CLICK_RATE, 0.6), CLICK_RATE)
            + quiet(0.9, CLICK_RATE),
        )
    },
)
def metronome_flashes_on_every_beat(w):
    beats = trim(w, doubled(w, load(w, "clique.wav"), 5), 0.0, 30.0)
    second = timeline(
        w, color_strip(w, (WHITE, NIGHT, NIGHT, NIGHT), (256, 144)), (256, 144), 4
    )
    movie(w, repeated(w, second, 30), 4.0, beats, "metronomo")

    def check(out):
        media = Media.returned(out, "metronomo")
        assert_thirty_seconds(
            media,
            fps=4,
            size=(256, 144),
            video="h264",
            audio="aac",
            rate=CLICK_RATE,
            channels=2,
        )
        sound = media.sound()
        for beat in range(SECONDS):
            assert (
                sound.amplitude(1000, beat + 0.02, beat + 0.08) > 0.3
            ), f"no click at {beat}s"
            assert (
                sound.rms(beat + 0.3, beat + 0.9) < 0.01
            ), f"noise after the click at {beat}s"
        for beat in (0, 14, 29):
            assert_pixel(media.frame(beat + 0.125), 128, 72, WHITE, 16)
            assert_pixel(media.frame(beat + 0.625), 128, 72, NIGHT, 16)

    return check


# ---------------------------------------------------------------- 2. stereo ping-pong

PING_RATE = 16_000


@video(
    "Pingue-pongue estéreo: um lá de 440 Hz (WAV) e 5 s de EmptyAudio são emendados com "
    "AudioConcat (after e before), reduzidos a um canal com SplitAudioChannels e juntados "
    "com JoinAudioChannels; o som troca de caixa a cada 5 s e acende a metade da tela do "
    "lado que toca.",
    fps=6.0,
    uploads={"la_440.wav": wav(PING_RATE, sine(440, 5.0, PING_RATE, 0.5))},
)
def stereo_ping_pong_between_speakers(w):
    tone, rest = load(w, "la_440.wav"), pause(w, 5.0, PING_RATE)
    tone_first, rest_first = concat(w, tone, rest), concat(w, tone, rest, "before")
    left = left_channel(w, concat(w, concat(w, tone_first, tone_first), tone_first))
    right = left_channel(w, concat(w, concat(w, rest_first, rest_first), rest_first))
    on_left = color_strip(w, (YELLOW, NAVY), (160, 180))
    on_right = color_strip(w, (NAVY, YELLOW), (160, 180))
    ten_seconds = timeline(w, stitch(w, on_left, on_right), (320, 180), 60, hold=30)
    movie(w, repeated(w, ten_seconds, 3), 6.0, stereo(w, left, right), "pingue_pongue")

    def check(out):
        media = Media.returned(out, "pingue_pongue")
        assert_thirty_seconds(
            media,
            fps=6,
            size=(320, 180),
            video="h264",
            audio="aac",
            rate=PING_RATE,
            channels=2,
        )
        sound = media.sound(channels=2)
        for part in range(6):
            playing, silent = (0, 1) if part % 2 == 0 else (1, 0)
            start, end = 5 * part + 0.5, 5 * part + 4.5
            assert (
                sound.amplitude(440, start, end, playing) > 0.35
            ), f"part {part} is not playing"
            assert (
                sound.rms(start, end, silent) < 0.01
            ), f"part {part} leaks to the other side"
            frame = media.frame(5 * part + 2.5)
            lit, dark = (80, 240) if playing == 0 else (240, 80)
            assert_pixel(frame, lit, 90, YELLOW, 16)
            assert_pixel(frame, dark, 90, NAVY, 16)

    return check


# ---------------------------------------------------------------- 3. volume staircase

STEP_RATE = 24_000
STEPS = (0, -6, -12, -18, -24, -30)
GRAYS = (0xF0F0F0, 0xB4B4B4, 0x878787, 0x5A5A5A, 0x373737, 0x1E1E1E)


@video(
    "Escada de volume: o mesmo mi de 660 Hz passa por seis AudioAdjustVolume (0 a -30 dB, "
    "de 6 em 6) e os trechos de 5 s são emendados; a tela escurece no mesmo passo.",
    fps=10.0,
    uploads={"mi_660.wav": wav(STEP_RATE, sine(660, 5.0, STEP_RATE, 0.8))},
)
def volume_staircase_dims_with_the_screen(w):
    tone = load(w, "mi_660.wav")
    steps = [louder(w, tone, step) for step in STEPS]
    sound = steps[0]
    for step in steps[1:]:
        sound = concat(w, sound, step)
    frames = timeline(w, color_strip(w, GRAYS, (256, 144)), (256, 144), 300, hold=50)
    movie(w, frames, 10.0, sound, "escada_de_volume")

    def check(out):
        media = Media.returned(out, "escada_de_volume")
        assert_thirty_seconds(
            media,
            fps=10,
            size=(256, 144),
            video="h264",
            audio="aac",
            rate=STEP_RATE,
            channels=2,
        )
        sound = media.sound()
        first = sound.amplitude(660, 0.5, 4.5)
        assert abs(first - 0.8) < 0.1, f"the first step plays at {first:.3f}, not 0.8"
        levels = [
            decibels(sound.rms(5 * part + 0.5, 5 * part + 4.5)) for part in range(6)
        ]
        drops = [round(a - b, 2) for a, b in zip(levels, levels[1:])]
        assert all(5.0 <= drop <= 7.0 for drop in drops), f"the steps drop {drops} dB"
        for part, gray in enumerate(GRAYS):
            assert_pixel(media.frame(5 * part + 2.5), 128, 72, gray, 10)

    return check


# ---------------------------------------------------------------- 4. chord and tremolo

CHORD_RATE = 16_000
C_MAJOR = (262, 330, 392)


@video(
    "Acorde e tremolo: dó, mi e sol (WAVs de 1 s) somados com AudioMerge add viram um "
    "acorde; multiplicado (AudioMerge multiply) por um LFO de 2 Hz, vira tremolo. Os dois "
    "são dobrados e cortados em 15 s; as listras da tela ficam firmes e depois pulsam no "
    "ritmo do tremolo.",
    fps=8.0,
    uploads={
        "do_262.wav": wav(CHORD_RATE, sine(262, 1.0, CHORD_RATE, 0.3)),
        "mi_330.wav": wav(CHORD_RATE, sine(330, 1.0, CHORD_RATE, 0.3)),
        "sol_392.wav": wav(CHORD_RATE, sine(392, 1.0, CHORD_RATE, 0.3)),
        "lfo_2hz.wav": wav(
            CHORD_RATE,
            [
                0.5 - 0.5 * math.cos(2 * math.pi * 2 * n / CHORD_RATE)
                for n in range(CHORD_RATE)
            ],
        ),
    },
)
def chord_then_tremolo(w):
    notes = [load(w, name) for name in ("do_262.wav", "mi_330.wav", "sol_392.wav")]
    triad = merge(w, merge(w, notes[0], notes[1], "add"), notes[2], "add")
    tremolo = merge(w, triad, load(w, "lfo_2hz.wav"), "multiply")
    plain = trim(w, doubled(w, triad, 4), 0.0, 15.0)
    pulsing = trim(w, doubled(w, tremolo, 4), 0.0, 15.0)
    size = (288, 160)
    stripes = color_strip(w, (RED, GREEN, BLUE), (96, 160))
    half = dimmed(w, stripes, size, 0.5)
    pulse = side_by_side(w, [dimmed(w, stripes, size, 0.85), half, stripes, half])
    steady = repeated(w, stripes, 120)
    beating = repeated(w, timeline(w, pulse, size, 4), 30)
    frames = timeline(w, stitch(w, steady, beating), size, 240)
    movie(w, frames, 8.0, concat(w, plain, pulsing), "acorde_e_tremolo")

    def check(out):
        media = Media.returned(out, "acorde_e_tremolo")
        assert_thirty_seconds(
            media,
            fps=8,
            size=size,
            video="h264",
            audio="aac",
            rate=CHORD_RATE,
            channels=2,
        )
        sound = media.sound()
        for frequency in C_MAJOR:
            assert (
                sound.amplitude(frequency, 1.0, 14.0) > 0.2
            ), f"{frequency} Hz is missing"
        for frequency in (300, 440):
            assert (
                sound.amplitude(frequency, 1.0, 14.0) < 0.03
            ), f"{frequency} Hz is not in the chord"
        steady_envelope = sound.envelope(2.0, 4.0, 0.05)
        assert min(steady_envelope) > 0.75 * max(
            steady_envelope
        ), "the chord is not steady"
        # The LFO starts at zero, so the tremolo peaks 0.25 s into each half second.
        for cycle in range(4, 28):
            peak = sound.rms(15.2 + cycle / 2, 15.3 + cycle / 2)
            trough = sound.rms(14.95 + cycle / 2, 15.05 + cycle / 2)
            assert trough < 0.3 * peak, f"no tremolo around {15 + cycle / 2}s"
        assert_pixel(media.frame(5.0625), 48, 80, RED, 16)
        assert_pixel(media.frame(16.3125), 48, 80, RED, 16)
        assert_pixel(
            media.frame(16.0625), 48, 80, tuple(round(c * 0.15) for c in rgb(RED)), 16
        )

    return check


# ---------------------------------------------------------------- 5. off the air

TV_RATE = 22_050
BARS = (0xC0C0C0, 0xC0C000, 0x00C0C0, 0x00C000, 0xC000C0, 0xC00000, 0x0000C0, 0x101010)


@video(
    "TV fora do ar: 10 s de chiado (ruído branco em WAV, dobrado e cortado) com estática "
    "animada do ImageAddNoise, depois 20 s de barras de cor com um tom de 1 kHz a -6 dB. "
    "Os dois lotes de quadros são emendados lado a lado e recortados de volta em sequência.",
    fps=10.0,
    uploads={
        "chiado.wav": wav(TV_RATE, hiss(1.0, TV_RATE, 0.5, seed=11)),
        "tom_1khz.wav": wav(TV_RATE, sine(1000, 1.0, TV_RATE, 0.5)),
    },
)
def tv_static_then_test_pattern(w):
    noise = trim(w, doubled(w, load(w, "chiado.wav"), 4), 0.0, 10.0)
    tone = louder(w, trim(w, doubled(w, load(w, "tom_1khz.wav"), 5), 0.0, 20.0), -6)
    snow = w.add_node(
        ImageAddNoise, image=solid(w, 320, 180, 0x808080, 100), seed=3, strength=1.0
    ).output["IMAGE"]
    bars = repeated(w, color_strip(w, BARS, (40, 180)), 100)
    frames = timeline(w, side_by_side(w, [snow, bars, bars]), (320, 180), 300)
    movie(w, frames, 10.0, concat(w, noise, tone), "fora_do_ar")

    def check(out):
        media = Media.returned(out, "fora_do_ar")
        assert_thirty_seconds(
            media,
            fps=10,
            size=(320, 180),
            video="h264",
            audio="aac",
            rate=TV_RATE,
            channels=2,
        )
        sound = media.sound()
        assert sound.rms(0.5, 9.5) > 0.15, "the static is too quiet"
        assert sound.amplitude(1000, 0.5, 9.5) < 0.05, "the static has a tone"
        assert sound.rms(9.0, 9.9) > 0.15, "the static stops before 10s"
        assert abs(sound.amplitude(1000, 10.1, 11.9) - 0.25) < 0.05, "no tone at 10s"
        assert abs(sound.amplitude(1000, 10.5, 29.5) - 0.25) < 0.05
        assert (
            abs(sound.rms(10.5, 29.5) - 0.25 / math.sqrt(2)) < 0.03
        ), "noise under the tone"
        first, second = media.frame(5.05), media.frame(5.15)
        assert spread(luminance(first)) > 25, "the static is flat"
        assert picture_difference(first, second) > 15, "the static does not move"
        bars_frame = media.frame(20.05)
        for index, color in enumerate(BARS):
            assert_pixel(bars_frame, 40 * index + 20, 90, color, 16)

    return check


# ---------------------------------------------------------------- 6. equalizer

EQ_RATE = 32_000
EQ_NOISE = hiss(1.0, EQ_RATE, 0.1, seed=5)
LOW, MID, HIGH = (100, 150), (950, 1000, 1050), (5000, 6000, 7000)
EQ_SETTINGS = (
    (
        "graves",
        RED,
        dict(
            low_gain_dB=12.0,
            low_freq=250,
            mid_gain_dB=-24.0,
            mid_freq=1000,
            mid_q=1.0,
            high_gain_dB=-24.0,
            high_freq=2000,
        ),
    ),
    (
        "medios",
        GREEN,
        dict(
            low_gain_dB=-24.0,
            low_freq=250,
            mid_gain_dB=12.0,
            mid_freq=1000,
            mid_q=2.0,
            high_gain_dB=-24.0,
            high_freq=4000,
        ),
    ),
    (
        "agudos",
        BLUE,
        dict(
            low_gain_dB=-24.0,
            low_freq=250,
            mid_gain_dB=-24.0,
            mid_freq=1000,
            mid_q=1.0,
            high_gain_dB=12.0,
            high_freq=4000,
        ),
    ),
)


@video(
    "Equalizador: o mesmo ruído branco (WAV de 1 s, dobrado e cortado em 10 s) passa por "
    "três AudioEqualizer3Band, que realçam graves, médios e agudos e cortam o resto; a "
    "tela fica vermelha, verde e azul, uma cor por faixa.",
    fps=12.0,
    uploads={"ruido_branco.wav": wav(EQ_RATE, EQ_NOISE)},
)
def equalizer_moves_the_noise_between_bands(w):
    noise = trim(w, doubled(w, load(w, "ruido_branco.wav"), 4), 0.0, 10.0)
    parts = [
        w.add_node(AudioEqualizer3Band, audio=noise, **gains).output["AUDIO"]
        for _, _, gains in EQ_SETTINGS
    ]
    colors = [color for _, color, _ in EQ_SETTINGS]
    frames = timeline(w, color_strip(w, colors, (256, 144)), (256, 144), 360, hold=120)
    movie(
        w,
        frames,
        12.0,
        concat(w, concat(w, parts[0], parts[1]), parts[2]),
        "equalizador",
    )

    def check(out):
        media = Media.returned(out, "equalizador")
        assert_thirty_seconds(
            media,
            fps=12,
            size=(256, 144),
            video="h264",
            audio="aac",
            rate=EQ_RATE,
            channels=2,
        )
        sound = media.sound()
        # The gains are measured against the noise that was sent, looped to 10 s.
        bands = {"graves": LOW, "medios": MID, "agudos": HIGH}
        source = Sound(EQ_RATE, [EQ_NOISE * 10])
        flat = {band: source.band(hz, 0.5, 9.5) for band, hz in bands.items()}
        for part, (name, color, _) in enumerate(EQ_SETTINGS):
            start, end = 10 * part + 0.5, 10 * part + 9.5
            gains = {
                band: decibels(max(sound.band(hz, start, end), 1e-9) / flat[band])
                for band, hz in bands.items()
            }
            shown = {band: round(gain, 1) for band, gain in gains.items()}
            assert 5.0 < gains[name] < 15.0, f"{name} is not boosted 12 dB: {shown}"
            assert all(
                gain < -12.0 for band, gain in gains.items() if band != name
            ), f"{name}: the other bands are not cut: {shown}"
            assert_pixel(media.frame(10 * part + 5.04), 128, 72, color, 16)

    return check


# ---------------------------------------------------------------- 7. musical scale

NOTES = {
    "do4": (262, 16_000),
    "re4": (294, 22_050),
    "mi4": (330, 32_000),
    "fa4": (349, 44_100),
    "sol4": (392, 48_000),
    "la4": (440, 24_000),
    "si4": (494, 11_025),
    "do5": (523, 8_000),
}
MELODY = ("do4", "re4", "mi4", "fa4", "sol4", "la4", "si4", "do5", "sol4", "do4")


@video(
    "Escala musical com taxas misturadas: oito notas de 1 s gravadas de 8 kHz a 48 kHz; "
    "cada uma é esticada para 3 s com AudioConcat e a melodia de dez notas é emendada, o "
    "que reamostra tudo para a maior taxa (48 kHz) sem mudar a afinação; cada nota tem a "
    "sua cor na tela.",
    fps=5.0,
    uploads={
        f"{name}.wav": wav(rate, sine(frequency, 1.0, rate, 0.5))
        for name, (frequency, rate) in NOTES.items()
    },
)
def musical_scale_across_sample_rates(w):
    held = {}
    for name in dict.fromkeys(MELODY):
        note = load(w, f"{name}.wav")
        held[name] = concat(w, concat(w, note, note), note)
    melody = held[MELODY[0]]
    for name in MELODY[1:]:
        melody = concat(w, melody, held[name])
    colors = hues(len(MELODY))
    frames = timeline(w, color_strip(w, colors, (320, 180)), (320, 180), 150, hold=15)
    movie(w, frames, 5.0, melody, "escala")

    def check(out):
        media = Media.returned(out, "escala")
        assert_thirty_seconds(
            media,
            fps=5,
            size=(320, 180),
            video="h264",
            audio="aac",
            rate=48_000,
            channels=2,
        )
        sound = media.sound()
        frequencies = [frequency for frequency, _ in NOTES.values()]
        for index, name in enumerate(MELODY):
            start, end = 3 * index + 0.3, 3 * index + 2.7
            expected = NOTES[name][0]
            assert (
                sound.dominant(frequencies, start, end) == expected
            ), f"note {index} is not {name}"
            assert (
                sound.amplitude(expected, start, end) > 0.3
            ), f"note {index} is too quiet"
            assert_pixel(media.frame(3 * index + 1.5), 160, 90, colors[index], 20)

    return check


# ---------------------------------------------------------------- 8. remix

CODE_RATE = 16_000
TIME_CODE = (300, 400, 500, 600, 700, 800, 900, 1000)


@video(
    "Remix com Video Slice: um vídeo de 40 s com um código de tempo sonoro (um tom por "
    "trecho de 5 s, num WAV) é cortado de 5 s a 35 s; GetVideoComponents tira os quadros, "
    "o fps e o áudio, os quadros são invertidos e o vídeo é recriado: som e cores chegam "
    "deslocados 5 s.",
    fps=10.0,
    uploads={
        "codigo_de_tempo.wav": wav(
            CODE_RATE,
            [value for tone in TIME_CODE for value in sine(tone, 5.0, CODE_RATE, 0.5)],
        )
    },
)
def video_slice_remix_shifts_the_time_code(w):
    colors = hues(len(TIME_CODE))
    frames = timeline(w, color_strip(w, colors, (256, 144)), (256, 144), 400, hold=50)
    source = w.add_node(
        CreateVideo, images=frames, fps=10.0, audio=load(w, "codigo_de_tempo.wav")
    ).output["VIDEO"]
    sliced = w.add_node(
        VideoSlice, video=source, start_time=5.0, duration=30.0, strict_duration=True
    ).output["VIDEO"]
    parts = w.add_node(GetVideoComponents, video=sliced).output
    negative = w.add_node(ImageInvert, image=parts["images"]).output["IMAGE"]
    remix = w.add_node(
        CreateVideo, images=negative, fps=parts["fps"], audio=parts["audio"]
    ).output["VIDEO"]
    w.add_node(
        SaveVideo,
        video=remix,
        filename_prefix="remix",
        format="mp4",
        format_codec="h264",
    )

    def check(out):
        media = Media.returned(out, "remix")
        assert_thirty_seconds(
            media,
            fps=10,
            size=(256, 144),
            video="h264",
            audio="aac",
            rate=CODE_RATE,
            channels=1,
            frame_slack=1,
        )
        sound = media.sound()
        for part in range(6):
            start, end = 5 * part + 0.5, 5 * part + 4.5
            expected = TIME_CODE[part + 1]
            assert (
                sound.dominant(TIME_CODE, start, end) == expected
            ), f"part {part} is not {expected} Hz"
            assert (
                sound.amplitude(expected, start, end) > 0.35
            ), f"{expected} Hz does not fill part {part}"
            assert_pixel(
                media.frame(5 * part + 2.55), 128, 72, inverted(colors[part + 1]), 20
            )

    return check


# ---------------------------------------------------------------- 9. binaural beat

BEAT_RATE = 44_100
SUNSET = (0xFFB347, 0xFF7F50, 0xE0558A, 0x9B4DCA, 0x4B3F9E, 0x1B1F4B)


@video(
    "Batimento binaural num WebM: 200 Hz à esquerda e 206 Hz à direita (dois WAVs mono) "
    "juntados com JoinAudioChannels, dobrados e cortados em 30 s; SaveVideo grava WebM com "
    "AV1 e Opus, que reamostra o som para 48 kHz. Somados, os canais batem 6 vezes por "
    "segundo; a tela passa pelas cores de um pôr do sol.",
    fps=12.0,
    uploads={
        "esquerda_200.wav": wav(BEAT_RATE, sine(200, 1.0, BEAT_RATE, 0.5)),
        "direita_206.wav": wav(BEAT_RATE, sine(206, 1.0, BEAT_RATE, 0.5)),
    },
)
def binaural_beat_in_webm(w):
    pair = stereo(w, load(w, "esquerda_200.wav"), load(w, "direita_206.wav"))
    frames = timeline(w, color_strip(w, SUNSET, (256, 144)), (256, 144), 360, hold=60)
    movie(
        w,
        frames,
        12.0,
        trim(w, doubled(w, pair, 5), 0.0, 30.0),
        "batimento",
        "webm",
        "av1",
    )

    def check(out):
        assert out.name("batimento").endswith(".webm")
        media = Media.returned(out, "batimento")
        assert "webm" in media.format["format_name"]
        assert_thirty_seconds(
            media,
            fps=12,
            size=(256, 144),
            video="av1",
            audio="opus",
            rate=48_000,
            channels=2,
        )
        sound = media.sound(channels=2)
        for second in (2, 15, 28):
            for channel, own, other in ((0, 200, 206), (1, 206, 200)):
                side = ("left", "right")[channel]
                assert (
                    sound.amplitude(own, second, second + 1, channel) > 0.35
                ), f"{own} Hz is missing on the {side} at {second}s"
                assert (
                    sound.amplitude(other, second, second + 1, channel) < 0.05
                ), f"{other} Hz leaks to the {side} at {second}s"
        mixed = media.sound().envelope(10.0, 11.0, 0.01)
        assert min(mixed) < 0.25 * max(mixed), "the channels do not beat together"
        for part, color in enumerate(SUNSET):
            assert_pixel(media.frame(5 * part + 2.54), 128, 72, color, 20)

    return check


# ---------------------------------------------------------------- 10. SDXL day at the lake

LAKE_RATE = 22_050
LAKE = (
    (
        "amanhecer",
        "sunrise over a calm mountain lake, soft pink light, morning mist, "
        "landscape photograph",
        (262, 330, 392),
    ),
    (
        "meio_dia",
        "bright midday over a calm mountain lake, deep blue sky, white clouds, "
        "landscape photograph",
        (349, 440, 523),
    ),
    (
        "por_do_sol",
        "golden sunset over a calm mountain lake, orange sky, long shadows, "
        "landscape photograph",
        (220, 262, 330),
    ),
    (
        "noite",
        "night over a calm mountain lake, starry sky, full moon, dark blue, "
        "landscape photograph",
        (196, 247, 294),
    ),
)
LAKE_SIZE = (512, 288)


@video(
    "Um dia no lago com SDXL: quatro cenas (amanhecer, meio-dia, pôr do sol, noite) a "
    "1344x768 com 25 passos, reduzidas e seguradas 7,5 s cada; a trilha toca um acorde "
    "por cena (WAVs de 1 s, dobrados e cortados), a noite 6 dB mais baixa, com vento "
    "(ruído suave) somado por AudioMerge.",
    fps=8.0,
    gpu=True,
    uploads={
        **{
            f"acorde_{name}.wav": wav(LAKE_RATE, chord(notes, 1.0, LAKE_RATE, 0.22))
            for name, _, notes in LAKE
        },
        "vento.wav": wav(LAKE_RATE, hiss(1.0, LAKE_RATE, 0.04, seed=9)),
    },
)
def sdxl_day_at_the_lake(w):
    model, clip, vae = sdxl(w)
    negative = prompt(w, clip, NEGATIVE + ", people, person, text")
    scenes, music = [], None
    for index, (name, text, _) in enumerate(LAKE):
        latent = ksample(
            w,
            model,
            prompt(w, clip, text),
            negative,
            empty_latent(w, 1344, 768),
            seed=21 + index,
            steps=25,
            cfg=7.0,
        )
        scenes.append(scale(w, decode(w, latent, vae), "area", *LAKE_SIZE))
        part = trim(w, doubled(w, load(w, f"acorde_{name}.wav"), 3), 0.0, 7.5)
        if name == "noite":
            part = louder(w, part, -6)
        music = part if music is None else concat(w, music, part)
    wind = trim(w, doubled(w, load(w, "vento.wav"), 5), 0.0, 30.0)
    frames = timeline(w, side_by_side(w, scenes), LAKE_SIZE, 240, hold=60)
    movie(w, frames, 8.0, merge(w, music, wind, "add"), "lago")

    def check(out):
        media = Media.returned(out, "lago")
        assert_thirty_seconds(
            media,
            fps=8,
            size=LAKE_SIZE,
            video="h264",
            audio="aac",
            rate=LAKE_RATE,
            channels=2,
        )
        sound = media.sound()
        every_note = {note for _, _, notes in LAKE for note in notes}
        for index, (name, _, notes) in enumerate(LAKE):
            start, end = 7.5 * index + 0.5, 7.5 * index + 7.0
            floor = 0.07 if name == "noite" else 0.15
            for note in notes:
                assert (
                    sound.amplitude(note, start, end) > floor
                ), f"{name}: {note} Hz is missing"
            for note in every_note - set(notes):
                assert (
                    sound.amplitude(note, start, end) < 0.03
                ), f"{name}: {note} Hz is extra"
        assert sound.rms(23.0, 29.5) < 0.7 * sound.rms(
            8.0, 14.5
        ), "the night is not quieter"
        pictures = [media.frame(7.5 * index + 3.76) for index in range(len(LAKE))]
        for (name, _, _), picture in zip(LAKE, pictures):
            assert spread(luminance(picture)) > 15, f"the {name} scene is flat"
        for first, second in zip(pictures, pictures[1:]):
            assert picture_difference(first, second) > 10, "two scenes are the same"
        mean = [
            sum(luminance(picture)) / len(luminance(picture)) for picture in pictures
        ]
        assert (
            mean[3] < mean[1]
        ), f"the night ({mean[3]:.0f}) is not darker than noon ({mean[1]:.0f})"

    return check


# ---------------------------------------------------------------- tests


def payload_of(case: Video) -> dict:
    workflow = Workflow()
    case.build(workflow)
    return sent(workflow)


def test_there_are_ten_videos_with_sound():
    payloads = [payload_of(case) for case in VIDEOS]

    assert len(VIDEOS) == 10
    assert len({case.name for case in VIDEOS}) == 10
    assert len({json.dumps(payload, sort_keys=True) for payload in payloads}) == 10
    for case, payload in zip(VIDEOS, payloads):
        saves = [node for node in payload.values() if node["class_type"] == "SaveVideo"]
        assert len(saves) == 1, case.name
        source = payload[saves[0]["inputs"]["video"][0]]
        assert (
            source["class_type"] == "CreateVideo" and "audio" in source["inputs"]
        ), case.name


@pytest.mark.parametrize("case", VIDEOS, ids=[case.name for case in VIDEOS])
def test_video_workflow_survives_a_reload(case):
    payload = payload_of(case)
    workflow = Workflow()

    workflow.load_workflow(json.loads(json.dumps(payload)))

    assert sent(workflow) == payload


@pytest.mark.parametrize("case", VIDEOS, ids=[case.name for case in VIDEOS])
def test_video_sends_every_sound_it_loads(case):
    loaded = {
        node["inputs"]["audio"]
        for node in payload_of(case).values()
        if node["class_type"] == "LoadAudio"
    }

    assert loaded == set(case.uploads)
    assert sum(len(data) for data in case.uploads.values()) < 1_500_000


@pytest.mark.parametrize("case", VIDEOS, ids=[case.name for case in VIDEOS])
def test_video_frame_rate_gives_thirty_seconds(case):
    payload = payload_of(case)
    (save,) = [node for node in payload.values() if node["class_type"] == "SaveVideo"]
    fps = payload[save["inputs"]["video"][0]]["inputs"]["fps"]

    assert (case.fps * SECONDS).is_integer()
    assert fps == case.fps or isinstance(fps, list)


def test_wav_keeps_rate_channels_and_samples():
    data = wav(8000, [0.0, 0.5, -0.5, 1.0], [0.25, -0.25, 0.0, -1.0])

    with wave.open(io.BytesIO(data)) as file:
        assert (file.getframerate(), file.getnchannels(), file.getnframes()) == (
            8000,
            2,
            4,
        )
        samples = array("h", file.readframes(4))
    if sys.byteorder == "big":
        samples.byteswap()
    assert list(samples) == [0, 8192, 16384, -8192, -16384, 0, 32767, -32767]


def test_sound_measures_tone_level_and_frequency():
    tone = Sound(8000, [sine(440, 1.0, 8000, 0.5)])

    assert abs(tone.amplitude(440, 0.0, 1.0) - 0.5) < 0.005
    assert tone.amplitude(660, 0.0, 1.0) < 0.005
    assert tone.dominant((330, 440, 550), 0.1, 0.9) == 440
    assert abs(tone.rms(0.0, 1.0) - 0.5 / math.sqrt(2)) < 0.002
    assert abs(decibels(tone.rms(0.0, 1.0)) - decibels(0.5 / math.sqrt(2))) < 0.05


def test_sound_band_and_envelope_follow_the_signal():
    rate = 8000
    noise = Sound(rate, [hiss(2.0, rate, 0.5, seed=1)])
    beating = Sound(
        rate,
        [[a + b for a, b in zip(sine(200, 1.0, rate, 0.5), sine(206, 1.0, rate, 0.5))]],
    )

    assert 0.0 < noise.band((500, 1000), 0.0, 2.0) < 0.05
    envelope = beating.envelope(0.0, 1.0, 0.01)
    assert min(envelope) < 0.1 * max(envelope)
    assert frange(0.0, 1.0, 0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.skipif(not TOOLS, reason="needs ffmpeg and ffprobe")
def test_media_measures_a_video_made_by_ffmpeg(tmp_path):
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0xE02020:s=64x48:r=4:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )

    media = Media(path.read_bytes(), ".mp4")

    assert media.size == (64, 48) and media.fps == 4 and media.frames == 8
    assert abs(media.seconds - 2.0) < 0.1
    assert (media.audio["codec_name"], int(media.audio["sample_rate"])) == (
        "aac",
        16000,
    )
    assert abs(stream_seconds(media.audio) - 2.0) < 0.1
    assert abs(media.sound().amplitude(440, 0.2, 1.8) - 0.125) < 0.02
    assert_pixel(media.frame(1.1), 32, 24, RED, 16)


def test_gallery_plays_the_videos_and_the_sent_sounds_with_audio(tmp_path):
    for name in ("lago_00001_.mp4", "batimento_00001_.webm", "entrada-clique.wav"):
        (tmp_path / name).write_bytes(b"")

    videos = [
        media_html(tmp_path, tmp_path / name)
        for name in ("lago_00001_.mp4", "batimento_00001_.webm")
    ]
    sound = media_html(tmp_path, tmp_path / "entrada-clique.wav")

    assert all("<video" in html and "muted" not in html for html in videos)
    assert "<audio" in sound and "entrada-clique.wav" in sound


@pytest.mark.skipif(not TOOLS, reason="needs ffmpeg and ffprobe")
def test_media_frame_is_the_one_shown_at_that_time(tmp_path):
    path = tmp_path / "flash.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=64x48:r=4:d=0.25",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x101828:s=64x48:r=4:d=0.75",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )

    media = Media(path.read_bytes(), ".mp4")

    assert media.frames == 4
    assert_pixel(media.frame(0.0), 32, 24, WHITE, 16)
    assert_pixel(media.frame(0.125), 32, 24, WHITE, 16)
    assert_pixel(media.frame(0.25), 32, 24, NIGHT, 16)
    assert_pixel(media.frame(0.875), 32, 24, NIGHT, 16)


def test_matroska_stream_length_comes_from_its_tag():
    assert stream_seconds(
        {"tags": {"DURATION": "00:00:30.004000000"}}
    ) == pytest.approx(30.004)
    assert stream_seconds({"duration": "29.97"}) == pytest.approx(29.97)


@pytest.mark.runpod
@pytest.mark.parametrize("case", VIDEOS, ids=[case.name for case in VIDEOS])
def test_video_with_sound_runs_on_comfy_ui(case, runpod_endpoint):
    if not TOOLS:
        pytest.skip("ffmpeg and ffprobe are needed to measure the videos")
    workflow = Workflow()
    check = case.build(workflow)

    run_and_record(
        runpod_endpoint,
        f"v{VIDEOS.index(case) + 1:02}-{case.name}",
        case.name,
        case.description,
        sent(workflow),
        timeout=GPU_TIMEOUT if case.gpu else TIMEOUT,
        uploads=case.uploads,
        check=check,
    )
