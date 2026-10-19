# comfy-ui-api-workflow

Build [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflows in Python
and export them in the API format that ComfyUI's `POST /prompt` endpoint runs.

- **665 ComfyUI nodes as Python classes**, from `KSampler` to Wan, LTXV, Flux,
  3D and partner API nodes.
- **Checked as you build**: input types, `min`/`max` ranges, combo choices,
  output names, socket types (no `LATENT` into a `CLIP` input) and cycles. A bad
  workflow fails in your code with a clear message, not in the ComfyUI queue.
- **Pipelines**: reusable groups of nodes with typed parameters and named
  outputs, added all or nothing.
- **Round trip**: load a workflow saved with ComfyUI's *Export (API)*, change it
  from Python, export it again.
- No dependencies. Python 3.10+.

## Installation

```bash
pip install git+https://github.com/fabioantonioastore/comfy-ui-api-workflow
```

## Quick start

```python
import json

from comfy_ui_api_workflow import Workflow, nodes

workflow = Workflow()
checkpoint = workflow.add_node(
    nodes.model.loaders.CheckpointLoaderSimple, ckpt_name="sd_xl_base_1.0.safetensors"
)
prompt = workflow.add_node(
    nodes.model.conditioning.CLIPTextEncode,
    text="a lighthouse at dusk",
    clip=checkpoint.output["CLIP"],
)
negative = workflow.add_node(
    nodes.model.conditioning.CLIPTextEncode, text="blurry", clip=checkpoint.output["CLIP"]
)
latent = workflow.add_node(
    nodes.model.latent.EmptyLatentImage, width=1024, height=1024, batch_size=1
)
sampler = workflow.add_node(
    nodes.model.sampling.KSampler,
    model=checkpoint.output["MODEL"],
    positive=prompt.output["CONDITIONING"],
    negative=negative.output["CONDITIONING"],
    latent_image=latent.output["LATENT"],
    sampler_name="dpmpp_2m",
    scheduler="karras",
    steps=25,
    cfg=7.0,
)
image = workflow.add_node(
    nodes.model.latent.VAEDecode, samples=sampler.output["LATENT"], vae=checkpoint.output["VAE"]
)
workflow.add_node(nodes.image.SaveImage, images=image.output["IMAGE"], filename_prefix="lighthouse")

print(json.dumps(workflow.get_comfy_ui_workflow_format(), indent=2))
```

The printed JSON is what ComfyUI's `/prompt` endpoint takes as `prompt`.

Mistakes are caught right away:

```python
from comfy_ui_api_workflow import Workflow, errors, nodes

workflow = Workflow()
latent = workflow.add_node(nodes.model.latent.EmptyLatentImage, width=512, height=512)

try:
    latent.width = 8
except errors.node.InvalidNodeParameterValue as error:
    print(error)  # width must be in closed range: [16, 16384]
```

## Documentation

- [User guide](docs/guide.md): workflows, links, validation, changing nodes,
  pipelines, loading and exporting, running on ComfyUI, errors.
- [API reference](docs/api.md): every public class, method, attribute and error.
- [Custom nodes](docs/custom_nodes.md): classes for nodes from custom node packs.
- [Node catalog](docs/nodes.md): every node with its inputs and outputs.
- [`examples/`](examples): runnable scripts with pipelines and nodes.

## Development

```bash
hatch test             # unit tests (RunPod tests are skipped)
hatch test --cover     # with coverage
hatch test -- --runpod # also send real jobs to the RunPod endpoint in .env (paid GPU)
```

## License

MIT
