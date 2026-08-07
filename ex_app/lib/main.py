import asyncio
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from threading import Event
from time import perf_counter, sleep

import niquests
import PIL.Image
from fastapi import FastAPI
from nc_py_api import NextcloudApp, NextcloudException
from nc_py_api.ex_app import (
    AppAPIAuthMiddleware,
    LogLvl,
    get_computation_device,
    persistent_storage,
    run_app,
    set_handlers,
)
from nc_py_api.ex_app.providers.task_processing import (
    ShapeDescriptor,
    ShapeType,
    TaskProcessingProvider,
    TaskType,
)
from niquests import codes
from niquests.exceptions import RequestException
from PIL import ImageDraw, ImageFont, PngImagePlugin
from stable_diffusion_cpp import StableDiffusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log(nc, level, content):
    logger.log((level + 1) * 10, content)
    if level < LogLvl.WARNING:
        return
    try:
        asyncio.run(nc.log(level, content))
    except Exception:
        logger.exception("Failed to log to Nextcloud")


TASKPROCESSING_PROVIDER_ID_BASIC = "text2image_flux:flux2_klein"
TASKPROCESSING_PROVIDER_ID_ENHANCED = "text2image_flux:flux2_klein_enhanced"
TASKPROCESSING_PROVIDER_ID_EDIT = "text2image_flux:flux2_klein_edit"
TASKPROCESSING_TYPE_EDIT = "core:image2image"
TASKPROCESSING_TYPE_EDIT_FALLBACK = "text2image_flux:image2image"
DEFAULT_SIZE = os.getenv("DEFAULT_SIZE", "1024x1024")
DEFAULT_GUIDANCE_SCALE = 1.0

DIFFUSION_MODEL_FILE = "flux-2-klein-4b-Q4_0.gguf"
VAE_MODEL_FILE = "flux2-vae.safetensors"
LLM_MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"

models_to_fetch = {
    # Pinned to repo commits so init downloads stay reproducible
    "https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/3b1f5a9dc3abb32238b053aeb3d823c30afdacbd/flux-2-klein-4b-Q4_0.gguf": {
        "save_path": os.path.join(persistent_storage(), DIFFUSION_MODEL_FILE),
    },
    "https://huggingface.co/Comfy-Org/flux2-dev/resolve/03d6521e6f6a47396b3f951cbea50f7e6c2f482e/split_files/vae/flux2-vae.safetensors": {
        "save_path": os.path.join(persistent_storage(), VAE_MODEL_FILE),
    },
    "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-Q4_K_M.gguf": {
        "save_path": os.path.join(persistent_storage(), LLM_MODEL_FILE),
    },
}

app_enabled = Event()
TRIGGER = Event()

WAIT_INTERVAL = 5
WAIT_INTERVAL_WITH_TRIGGER = 5 * 60
WATERMARK_COMMENT = "Generated using Artificial Intelligence"

PROVIDER_IDS = [
    TASKPROCESSING_PROVIDER_ID_BASIC,
    TASKPROCESSING_PROVIDER_ID_ENHANCED,
    TASKPROCESSING_PROVIDER_ID_EDIT,
]
TASK_TYPES = ["core:text2image", TASKPROCESSING_TYPE_EDIT]


def load_model() -> StableDiffusion:
    storage = persistent_storage()
    return StableDiffusion(
        diffusion_model_path=os.path.join(storage, DIFFUSION_MODEL_FILE),
        llm_path=os.path.join(storage, LLM_MODEL_FILE),
        vae_path=os.path.join(storage, VAE_MODEL_FILE),
        offload_params_to_cpu=True,
        diffusion_flash_attn=True,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global TASKPROCESSING_TYPE_EDIT, TASK_TYPES
    set_handlers(
        APP,
        enabled_handler,
        trigger_handler=trigger_handler,
        models_to_fetch=models_to_fetch,
    )
    nc = NextcloudApp()
    # Use a custom image2image task type on Nextcloud <= 35
    if nc.srv_version.get("major") <= 35:
        TASKPROCESSING_TYPE_EDIT = TASKPROCESSING_TYPE_EDIT_FALLBACK
        TASK_TYPES = ["core:text2image", TASKPROCESSING_TYPE_EDIT]
    if nc.enabled_state:
        app_enabled.set()
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)
APP.add_middleware(AppAPIAuthMiddleware)


def schedule_prompt_improvement_and_wait(nc: NextcloudApp, original_prompt: str) -> str:
    if original_prompt.strip() == "":
        return original_prompt
    try:
        data = nc.ocs(
            "POST",
            "/ocs/v1.php/taskprocessing/schedule?format=json",
            headers={"OCS-APIRequest": "true"},
            json={
                "input": {"input": original_prompt},
                "type": "core:text2text",
                "appId": os.environ["APP_ID"],
            },
        )
    except RequestException as e:
        raise RuntimeError(f"Failed to schedule prompt improvement task: {e}") from e

    task_id = data.get("task", {}).get("id")

    if not isinstance(task_id, int):
        raise RuntimeError(f"Unexpected schedule response: {data!r}")

    task = {"id": task_id, "status": "STATUS_SCHEDULED", "output": None}
    i = 0
    while task.get("status") not in ("STATUS_SUCCESSFUL", "STATUS_FAILED") and i < 60 * 6:
        if i < 60 * 3:
            sleep(5)
            i += 1
        else:
            sleep(10)
            i += 2

        try:
            response = nc.ocs("GET", f"/ocs/v1.php/taskprocessing/task/{task_id}")
        except (
            niquests.exceptions.ConnectionError,
            niquests.exceptions.Timeout,
        ) as e:
            log(nc, LogLvl.WARNING, f"Ignored error during task polling: {e}")
            sleep(5)
            i += 1
            continue
        except NextcloudException as e:
            if getattr(e, "status_code", None) == niquests.codes.too_many_requests:
                log(nc, LogLvl.WARNING, "Rate limited during task polling, waiting 10s before retrying")
                sleep(10)
                i += 2
                continue
            raise RuntimeError("Failed to poll Nextcloud TaskProcessing task") from e

        task = (response or {}).get("task", task)
        log(nc, LogLvl.INFO, f"Task poll ({i * 5}s) response: {task}")

    if task.get("status") == "STATUS_SUCCESSFUL":
        output = (task.get("output") or {}).get("output")
        if isinstance(output, str) and output.strip():
            return output
        raise RuntimeError(f"Prompt improvement returned empty output: {task!r}")
    if task.get("status") == "STATUS_FAILED":
        raise RuntimeError(f"Prompt improvement failed: {task!r}")
    raise RuntimeError("Prompt improvement timed out")


def start_bg_task():
    t = threading.Thread(target=background_thread_task, daemon=True)
    t.start()


def background_thread_task():
    nc = NextcloudApp()
    while not app_enabled.is_set():
        sleep(5)

    pipe = load_model()
    log(nc, LogLvl.INFO, f"Model loaded (device hint: {get_computation_device() or 'CPU'})")

    while True:
        if not app_enabled.is_set() or pipe is None:
            sleep(30)
            continue
        try:
            next_task = nc.providers.task_processing.next_task(PROVIDER_IDS, TASK_TYPES)
            if next_task is None or "task" not in next_task:
                wait_for_task()
                continue
            task = next_task.get("task")
            provider_id = next_task.get("provider", {}).get("name")
        except Exception as e:
            log(nc, LogLvl.ERROR, str(e))
            wait_for_task(30)
            continue
        try:
            handle_task(nc, task, provider_id, pipe)
        except Exception as e:
            log(nc, LogLvl.ERROR, str(e))
            try:
                nc.providers.task_processing.report_result(task["id"], None, str(e))
            except Exception:
                pass
            wait_for_task(30)


def download_task_file(nc: NextcloudApp, task_id: int, file_id: int) -> PIL.Image.Image:
    session = nc._session
    session.init_adapter()
    path = f"/ocs/v2.php/taskprocessing/tasks_provider/{task_id}/file/{file_id}"
    info = f"request: GET {path}"
    response = session.adapter.request("GET", path, stream=True)
    status_code = response.status_code
    if 996 <= status_code <= 999:
        if status_code == 996:
            phrase = "Server error"
        elif status_code == 997:
            phrase = "Unauthorised"
        elif status_code == 998:
            phrase = "Not found"
        else:
            phrase = "Unknown error"
        raise NextcloudException(status_code, reason=phrase, info=info)
    if status_code >= 400:
        raise NextcloudException(status_code, reason=codes(status_code).phrase, info=info)
    buf = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            buf.write(chunk)
    buf.seek(0)
    return PIL.Image.open(buf).convert("RGB")


def upload_result_image(nc: NextcloudApp, task_id: int, image: PIL.Image.Image) -> int:
    mark_image(image)
    png_stream = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", WATERMARK_COMMENT)
    image.save(png_stream, format="PNG", pnginfo=metadata)
    png_stream.seek(0)
    return nc.providers.task_processing.upload_result_file(task_id, png_stream)


def parse_size(size: str) -> tuple[int, int]:
    width, height = size.split("x")
    return int(width), int(height)


def handle_task(nc: NextcloudApp, task: dict, provider_id: str, pipe: StableDiffusion):
    log(nc, LogLvl.INFO, f"Next task: {task['id']}")
    task_type = task.get("type")
    if (
        provider_id == TASKPROCESSING_PROVIDER_ID_EDIT
        or task_type in ("core:image2image", TASKPROCESSING_TYPE_EDIT_FALLBACK)
    ):
        handle_image2image(nc, task, pipe)
    else:
        handle_text2image(nc, task, provider_id, pipe)


def handle_text2image(nc: NextcloudApp, task: dict, provider_id: str, pipe: StableDiffusion):
    number_of_images = task.get("input", {}).get("numberOfImages") or 1
    if number_of_images > 12 or number_of_images < 1:
        nc.providers.task_processing.report_result(task["id"], None, "numberOfImages is out of bounds")
        return

    log(nc, LogLvl.INFO, "generating image")
    time_start = perf_counter()
    original_prompt = task.get("input", {}).get("input")
    prompt = original_prompt
    progress = 0
    nc.set_user(task["userId"])

    result = {}

    if provider_id == TASKPROCESSING_PROVIDER_ID_ENHANCED:
        if task.get("userId") is None:
            log(nc, LogLvl.WARNING, "userId is None skipping prompt improvement")
        else:
            transcript = (
                "Please refine the following image-generation prompt to help a text-to-image model create a stunning, visually captivating, and coherent image. "
                "Where appropriate, enrich the prompt with specific visual details such as subject, composition, lighting, atmosphere, and artistic style. "
                "Preserve the original intent. Return ONLY the improved prompt as a single line, without any preamble, explanation, or quotes.\n\n"
                "Original prompt:\n"
                + original_prompt
            )
            try:
                log(nc, LogLvl.INFO, "scheduling prompt improvement")
                prompt = schedule_prompt_improvement_and_wait(nc, transcript)
                NextcloudApp().providers.task_processing.set_progress(task.get("id"), 25)
                progress = 25
                result["enhanced_prompt"] = prompt
                log(nc, LogLvl.INFO, "prompt improvement successful")
            except Exception as e:
                log(nc, LogLvl.WARNING, f"prompt improvement failed, using original prompt: {e}")

    log(nc, LogLvl.INFO, f"prompt: {prompt}")

    size = task.get("input", {}).get("size") or DEFAULT_SIZE
    width, height = parse_size(size)
    inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", 4))

    def on_progress(step: int, steps: int, _time: float):
        if steps <= 0:
            return
        NextcloudApp().providers.task_processing.set_progress(
            task.get("id"), progress + (step + 1) / steps * (100 - progress)
        )

    images = pipe.generate_image(
        prompt=prompt,
        height=height,
        width=width,
        cfg_scale=DEFAULT_GUIDANCE_SCALE,
        sample_steps=inference_steps,
        sample_method="euler",
        batch_count=number_of_images,
        progress_callback=on_progress,
    )

    img_ids = [upload_result_image(nc, task.get("id"), image) for image in images]

    log(nc, LogLvl.INFO, f"image generated: {perf_counter() - time_start}s")
    result["images"] = img_ids
    NextcloudApp().providers.task_processing.report_result(task["id"], result)


def handle_image2image(nc: NextcloudApp, task: dict, pipe: StableDiffusion):
    task_input = task.get("input", {})
    file_ids = task_input.get("input") or []
    prompt = task_input.get("prompt")

    if not isinstance(file_ids, list) or len(file_ids) == 0:
        nc.providers.task_processing.report_result(task["id"], None, "input images are required")
        return
    if not isinstance(prompt, str) or not prompt.strip():
        nc.providers.task_processing.report_result(task["id"], None, "prompt is required")
        return

    log(nc, LogLvl.INFO, "editing image")
    time_start = perf_counter()
    nc.set_user(task["userId"])

    ref_images = [download_task_file(nc, task["id"], int(file_id)) for file_id in file_ids]
    log(nc, LogLvl.INFO, f"loaded {len(ref_images)} reference image(s)")

    size = task_input.get("size") or DEFAULT_SIZE
    width, height = parse_size(size)

    inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", 4))
    log(nc, LogLvl.INFO, f"prompt: {prompt}")

    def on_progress(step: int, steps: int, _time: float):
        if steps <= 0:
            return
        NextcloudApp().providers.task_processing.set_progress(task.get("id"), (step + 1) / steps * 100)

    images = pipe.generate_image(
        prompt=prompt,
        ref_images=ref_images,
        height=height,
        width=width,
        cfg_scale=DEFAULT_GUIDANCE_SCALE,
        sample_steps=inference_steps,
        sample_method="euler",
        batch_count=1,
        progress_callback=on_progress,
    )

    if not images:
        nc.providers.task_processing.report_result(task["id"], None, "image editing produced no output")
        return

    output_id = upload_result_image(nc, task.get("id"), images[0])
    log(nc, LogLvl.INFO, f"image edited: {perf_counter() - time_start}s")
    NextcloudApp().providers.task_processing.report_result(task["id"], {"output": output_id})


async def enabled_handler(enabled: bool, nc: NextcloudApp) -> str:
    global TASKPROCESSING_TYPE_EDIT, TASK_TYPES
    print(f"enabled={enabled}")
    if enabled:
        await nc.log(LogLvl.WARNING, f"Enabled: {nc.app_cfg.app_name}")
        await nc.providers.task_processing.register(
            TaskProcessingProvider(
                id=TASKPROCESSING_PROVIDER_ID_BASIC,
                name="Nextcloud Local Image Generation: Flux 2 Klein 4B",
                task_type="core:text2image",
                expected_runtime=60,
                optional_input_shape=[
                    ShapeDescriptor(
                        name="size",
                        description=(
                            "Optional. The size of the generated images. "
                            "Must be in 1024x1024 format. Default is 1024x1024"
                        ),
                        shape_type=ShapeType.TEXT,
                    ),
                ],
                input_shape_defaults={"size": DEFAULT_SIZE, "numberOfImages": 1},
            )
        )
        await nc.providers.task_processing.register(
            TaskProcessingProvider(
                id=TASKPROCESSING_PROVIDER_ID_ENHANCED,
                name="Nextcloud Local Image Generation: Flux 2 Klein 4B (Enhanced)",
                task_type="core:text2image",
                expected_runtime=80,
                optional_input_shape=[
                    ShapeDescriptor(
                        name="size",
                        description=(
                            "Optional. The size of the generated images. "
                            "Must be in 1024x1024 format. Default is 1024x1024"
                        ),
                        shape_type=ShapeType.TEXT,
                    ),
                ],
                input_shape_defaults={"size": DEFAULT_SIZE, "numberOfImages": 1},
                optional_output_shape=[
                    ShapeDescriptor(
                        name="enhanced_prompt",
                        description="The enhanced prompt used for the image generation.",
                        shape_type=ShapeType.TEXT,
                    ),
                ],
            )
        )
        new_task_type = None
        TASKPROCESSING_TYPE_EDIT = "core:image2image"
        # Use a custom image2image task type on Nextcloud <= 35
        if (await nc.srv_version).get("major") <= 35:
            await nc.log(LogLvl.INFO, f"Creating custom image2image task type for {nc.app_cfg.app_name}")
            new_task_type = TaskType(
                id=TASKPROCESSING_TYPE_EDIT_FALLBACK,
                name="Edit image",
                description="Edit an image based on a text description of the changes",
                input_shape=[
                    ShapeDescriptor(
                        name="input",
                        description="The images to edit",
                        shape_type=ShapeType.LIST_OF_IMAGES,
                    ),
                    ShapeDescriptor(
                        name="prompt",
                        description="Describe the changes you want to make to the image",
                        shape_type=ShapeType.TEXT,
                    ),
                ],
                output_shape=[
                    ShapeDescriptor(
                        name="output",
                        description="The edited image",
                        shape_type=ShapeType.IMAGE,
                    ),
                ],
            )
            TASKPROCESSING_TYPE_EDIT = TASKPROCESSING_TYPE_EDIT_FALLBACK
        TASK_TYPES = ["core:text2image", TASKPROCESSING_TYPE_EDIT]
        await nc.providers.task_processing.register(
            TaskProcessingProvider(
                id=TASKPROCESSING_PROVIDER_ID_EDIT,
                name="Nextcloud Local Image Editing: Flux 2 Klein 4B",
                task_type=TASKPROCESSING_TYPE_EDIT,
                expected_runtime=60,
                optional_input_shape=[
                    ShapeDescriptor(
                        name="size",
                        description=(
                            "Optional. The size of the edited image. "
                            "Must be in 1024x1024 format. Default is 1024x1024"
                        ),
                        shape_type=ShapeType.TEXT,
                    ),
                ],
                input_shape_defaults={"size": DEFAULT_SIZE},
            ),
            new_task_type,
        )
        app_enabled.set()
    else:
        await nc.providers.task_processing.unregister(TASKPROCESSING_PROVIDER_ID_BASIC, True)
        await nc.providers.task_processing.unregister(TASKPROCESSING_PROVIDER_ID_ENHANCED, True)
        await nc.providers.task_processing.unregister(TASKPROCESSING_PROVIDER_ID_EDIT, True)
        await nc.log(LogLvl.WARNING, f"Disabled {nc.app_cfg.app_name}")
        app_enabled.clear()
    return ""


def trigger_handler(_provider_id: str):
    # This will only get called on Nextcloud 33+
    TRIGGER.set()


def wait_for_task(interval=None):
    global WAIT_INTERVAL
    if interval is None:
        interval = WAIT_INTERVAL
    if TRIGGER.wait(timeout=interval):
        WAIT_INTERVAL = WAIT_INTERVAL_WITH_TRIGGER
    TRIGGER.clear()


def mark_image(image: PIL.Image.Image):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = WATERMARK_COMMENT

    img_width, img_height = image.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 10
    x = img_width - text_width - margin
    y = img_height - text_height - margin

    draw.text((x, y), text, fill="white", font=font, stroke_width=1, stroke_fill="black")


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
