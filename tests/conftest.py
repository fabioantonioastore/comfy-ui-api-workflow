import base64
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

import pytest

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
# Files returned by the RunPod jobs, one folder per test; ignored by git.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "runpod_outputs"
RUNPOD_SETTINGS = ("RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID")


@pytest.fixture
def ksampler_inputs():
    return {
        "model": ("CheckpointLoaderSimple_0", 0),
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": ("CLIPTextEncode_1", 0),
        "negative": ("CLIPTextEncode_2", 0),
        "latent_image": ("EmptyLatentImage_3", 0),
    }


def pytest_addoption(parser):
    parser.addoption(
        "--runpod",
        action="store_true",
        help="send real jobs to the RunPod ComfyUI endpoint in .env (paid GPU time)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "runpod: sends real jobs to the RunPod endpoint; needs --runpod"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runpod"):
        return
    skip = pytest.mark.skip(reason="sends real jobs to RunPod; run with --runpod")
    for item in items:
        if "runpod" in item.keywords:
            item.add_marker(skip)


class RunPodEndpoint:
    """Runs workflows on a RunPod serverless endpoint that serves ComfyUI."""

    def __init__(self, endpoint: str, api_key: str) -> None:
        if not endpoint.startswith("http"):
            endpoint = f"https://api.runpod.ai/v2/{endpoint}"
        self.url = endpoint.rstrip("/")
        self.__api_key = api_key

    def run(
        self,
        workflow: dict,
        timeout: float = 600,
        images: dict[str, bytes] | None = None,
    ) -> dict:
        job_input: dict = {"workflow": workflow}
        if images:
            # worker-comfyui uploads these to ComfyUI's input folder before the run.
            job_input["images"] = [
                {"name": name, "image": base64.b64encode(data).decode()}
                for name, data in images.items()
            ]
        job = self.__request("POST", "/run", {"input": job_input})
        deadline = time.monotonic() + timeout
        while job["status"] in ("IN_QUEUE", "IN_PROGRESS"):
            if time.monotonic() > deadline:
                pytest.fail(f"RunPod job {job['id']} did not finish in {timeout}s")
            time.sleep(2)
            job = self.__request("GET", f"/status/{job['id']}")
        return job

    def __request(self, method: str, path: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self.__api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())


def read_env_file(path: Path) -> dict[str, str]:
    # Accepts both "NAME=value" and "NAME: value" lines.
    if not path.exists():
        return {}
    settings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*[:=]\s*(.*)", line)
        if match and not line.lstrip().startswith("#"):
            settings[match.group(1)] = match.group(2).strip().strip("'\"")
    return settings


class RunPodJobs:
    """Runs a test's jobs and keeps every file the worker returns in its folder."""

    def __init__(self, endpoint: RunPodEndpoint, folder: Path) -> None:
        self.endpoint = endpoint
        self.folder = folder

    def run(self, workflow: dict, timeout: float = 600) -> dict:
        job = self.endpoint.run(workflow, timeout)
        for item in (job.get("output") or {}).get("images", []):
            if item.get("type") == "base64":
                self.folder.mkdir(parents=True, exist_ok=True)
                # Only the base name, so a file name from the server stays inside.
                target = self.folder / Path(item["filename"]).name
                target.write_bytes(base64.b64decode(item["data"]))
        return job


@pytest.fixture(scope="session")
def runpod_endpoint():
    settings = {**read_env_file(ENV_FILE), **os.environ}
    missing = [name for name in RUNPOD_SETTINGS if not settings.get(name)]
    if missing:
        pytest.fail(f"--runpod needs {', '.join(missing)} in .env or the environment")
    return RunPodEndpoint(settings["RUNPOD_ENDPOINT_ID"], settings["RUNPOD_API_KEY"])


@pytest.fixture
def runpod(runpod_endpoint, request):
    folder = OUTPUT_DIR / request.node.name
    shutil.rmtree(folder, ignore_errors=True)
    return RunPodJobs(runpod_endpoint, folder)
