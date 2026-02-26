import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Return value from sequence-like output, with mapping fallback."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def _safe_get_index(obj: Union[Sequence, Mapping], index: int, default: Any = None) -> Any:
    try:
        return get_value_at_index(obj, index)
    except (IndexError, KeyError, TypeError):
        return default


def find_path(name: str, path: str = None) -> str:
    """Recursively search parent directories for `name`."""
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py, trying utils.")
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Initialize ComfyUI server + custom nodes for NODE_CLASS_MAPPINGS usage."""
    import asyncio

    import execution
    from nodes import init_extra_nodes

    comfy_path = find_path("ComfyUI")
    if comfy_path is None:
        raise RuntimeError("ComfyUI directory not found.")
    sys.path.insert(0, comfy_path)
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())


_NODES_READY = False
NODE_CLASS_MAPPINGS = None


@dataclass
class Sam2Runtime:
    vhs_loadvideo: Any
    vhs_selectimages: Any
    vhs_videoinfo: Any
    pointseditor: Any
    sam2videosegmentationaddpoints: Any
    sam2videosegmentation: Any
    maskfix: Any
    sam2segmentation: Any
    sam2_model: Any


def _ensure_nodes_initialized() -> None:
    global _NODES_READY, NODE_CLASS_MAPPINGS
    if _NODES_READY:
        return
    import_custom_nodes()
    try:
        from nodes import NODE_CLASS_MAPPINGS as mappings
    except ImportError as exc:
        raise RuntimeError(
            "ComfyUI custom nodes are not available. Ensure ComfyUI and node dependencies are installed."
        ) from exc
    NODE_CLASS_MAPPINGS = mappings
    _NODES_READY = True


def init_runtime(
    model_name: str = "sam2.1_hiera_large.safetensors",
    device: str = "cuda",
    precision: str = "fp16",
) -> Sam2Runtime:
    """Create node instances and preload the SAM2 video model once."""
    _ensure_nodes_initialized()
    with torch.inference_mode():
        download_node = NODE_CLASS_MAPPINGS["DownloadAndLoadSAM2Model"]()
        model_out = download_node.loadmodel(
            model=model_name,
            segmentor="video",
            device=device,
            precision=precision,
        )
    return Sam2Runtime(
        vhs_loadvideo=NODE_CLASS_MAPPINGS["VHS_LoadVideo"](),
        vhs_selectimages=NODE_CLASS_MAPPINGS["VHS_SelectImages"](),
        vhs_videoinfo=NODE_CLASS_MAPPINGS["VHS_VideoInfo"](),
        pointseditor=NODE_CLASS_MAPPINGS["PointsEditor"](),
        sam2videosegmentationaddpoints=NODE_CLASS_MAPPINGS["Sam2VideoSegmentationAddPoints"](),
        sam2videosegmentation=NODE_CLASS_MAPPINGS["Sam2VideoSegmentation"](),
        maskfix=NODE_CLASS_MAPPINGS["MaskFix+"](),
        sam2segmentation=NODE_CLASS_MAPPINGS["Sam2Segmentation"](),
        sam2_model=get_value_at_index(model_out, 0),
    )


def load_video_state(runtime: Sam2Runtime, video_path: str) -> dict[str, Any]:
    """Load a user video and return frame/video metadata state."""
    if not video_path:
        raise ValueError("video_path is required.")
    with torch.inference_mode():
        load_out = runtime.vhs_loadvideo.load_video(
            video=video_path,
            force_rate=0,
            custom_width=0,
            custom_height=0,
            frame_load_cap=0,
            skip_first_frames=0,
            select_every_nth=1,
            format="AnimateDiff",
            unique_id=0,
        )
        info_out = runtime.vhs_videoinfo.get_video_info(video_info=get_value_at_index(load_out, 3))

    video_state = {
        "video_path": video_path,
        "loaded_video": load_out,
        "frames": get_value_at_index(load_out, 0),
        "video_info": info_out,
        "frame_count": _safe_get_index(info_out, 10, 0),
        "width": _safe_get_index(info_out, 8, 0),
        "height": _safe_get_index(info_out, 9, 0),
    }
    return video_state


def _build_points(
    runtime: Sam2Runtime,
    width: int,
    height: int,
    positive_points: list[dict[str, float]],
    negative_points: list[dict[str, float]],
) -> tuple[Any, Any]:
    # PointsEditor expects array-like coordinate groups with consistent dimensions.
    # When one side is empty, use an out-of-frame placeholder to keep shape stable.
    safe_positive = positive_points if positive_points else [{"x": -1.0, "y": -1.0}]
    safe_negative = negative_points if negative_points else [{"x": -1.0, "y": -1.0}]

    with torch.inference_mode():
        points_out = runtime.pointseditor.pointdata(
            points_store="",
            coordinates=json.dumps(safe_positive),
            neg_coordinates=json.dumps(safe_negative),
            bbox_store="[{}]",
            bboxes="[{}]",
            bbox_format="xyxy",
            width=int(width),
            height=int(height),
            normalize=False,
        )
    return get_value_at_index(points_out, 0), get_value_at_index(points_out, 1)


def get_frame_image(runtime: Sam2Runtime, video_state: dict[str, Any], frame_index: int) -> Any:
    """Select a single frame tensor/image from loaded video frames."""
    with torch.inference_mode():
        frame_out = runtime.vhs_selectimages.select(
            indexes=str(int(frame_index)),
            err_if_missing=True,
            err_if_empty=True,
            image=video_state["frames"],
        )
    return get_value_at_index(frame_out, 0)


def _to_uint8_frame_sequence(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    while arr.ndim > 4:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D frame sequence, got shape {arr.shape}.")
    if arr.shape[-1] not in (1, 3, 4) and arr.shape[1] in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_uint8_mask_sequence(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    while arr.ndim > 4:
        arr = arr[0]
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, ...]
        else:
            arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D mask sequence, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _encode_raw_video(raw_frames: np.ndarray, input_pix_fmt: str, fps: float, output_path: str, codec_args: list[str]) -> str:
    if raw_frames.ndim != 4:
        raise ValueError("raw_frames must be [N, H, W, C].")
    n, h, w, _ = raw_frames.shape
    if n == 0:
        raise ValueError("Cannot write video with zero frames.")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        input_pix_fmt,
        "-s",
        f"{w}x{h}",
        "-r",
        f"{max(float(fps), 1.0):.3f}",
        "-i",
        "-",
        "-an",
        *codec_args,
        output_path,
    ]
    proc = subprocess.run(
        cmd,
        input=raw_frames.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    _ = proc
    return output_path


def _parse_ffmpeg_ratio(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "/" in text:
        try:
            num, den = text.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            fps = float(num) / den_f
            return fps if fps > 0 else None
        except ValueError:
            return None
    try:
        fps = float(text)
        return fps if fps > 0 else None
    except ValueError:
        return None


def _infer_input_video_fps(video_state: dict[str, Any], default_fps: float = 24.0) -> float:
    video_path = video_state.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        return default_fps

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        for line in proc.stdout.splitlines():
            fps = _parse_ffmpeg_ratio(line)
            if fps is not None:
                return max(fps, 1.0)
    except Exception:
        pass
    return default_fps


def _export_masked_rgba_video(video_state: dict[str, Any], mask_sequence: Any, fps: float = 24.0) -> str:
    frames = _to_uint8_frame_sequence(video_state["frames"])
    masks = _to_uint8_mask_sequence(mask_sequence)
    total = min(len(frames), len(masks))
    if total == 0:
        raise ValueError("No frames or masks available to export.")

    frames = frames[:total]
    masks = masks[:total]
    if masks.shape[1] != frames.shape[1] or masks.shape[2] != frames.shape[2]:
        raise ValueError(
            f"Mask/frame size mismatch. mask={masks.shape[1:3]} frame={frames.shape[1:3]}. "
            "Ensure tracking mask resolution matches video frame resolution."
        )
    # Use a stricter threshold so low-value soft masks don't keep full-frame visibility.
    mask_bin = (masks >= 128).astype(np.uint8)
    rgb_masked = (frames * mask_bin[..., None]).astype(np.uint8)

    output_dir = os.path.join(tempfile.gettempdir(), "sam2_rgba_outputs")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"sam2_masked_{uuid.uuid4().hex}")

    # Prefer standard MP4 codecs for maximum compatibility when downloaded.
    attempts: list[tuple[str, np.ndarray, str, list[str]]] = [
        (
            f"{base}.mp4",
            rgb_masked,
            "rgb24",
            ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
        (
            f"{base}.mp4",
            rgb_masked,
            "rgb24",
            ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"],
        ),
    ]

    errors: list[str] = []
    for out_path, raw_frames, in_pix_fmt, codec_args in attempts:
        try:
            return _encode_raw_video(
                raw_frames=raw_frames,
                input_pix_fmt=in_pix_fmt,
                fps=fps,
                output_path=out_path,
                codec_args=codec_args,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to export masked video but was not found.") from exc
        except subprocess.CalledProcessError as exc:
            errors.append(exc.stderr.decode(errors="ignore"))

    raise RuntimeError("ffmpeg could not encode masked video with available encoders.\n" + "\n---\n".join(errors))


def run_single_image(
    runtime: Sam2Runtime,
    video_state: dict[str, Any],
    frame_index: int,
    positive_points: list[dict[str, float]],
    negative_points: list[dict[str, float]],
    apply_maskfix: bool = True,
) -> dict[str, Any]:
    """Segment currently selected frame only (mode='single_image')."""
    if not positive_points and not negative_points:
        raise ValueError("At least one point is required for segmentation.")

    frame_image = get_frame_image(runtime, video_state, frame_index)
    pos_coords, neg_coords = _build_points(
        runtime,
        width=video_state.get("width", 0),
        height=video_state.get("height", 0),
        positive_points=positive_points,
        negative_points=negative_points,
    )
    with torch.inference_mode():
        seg_out = runtime.sam2segmentation.segment(
            keep_model_loaded=True,
            individual_objects=False,
            sam2_model=runtime.sam2_model,
            image=frame_image,
            coordinates_positive=pos_coords,
            coordinates_negative=neg_coords,
        )
        mask = get_value_at_index(seg_out, 0)
        if apply_maskfix:
            mask = get_value_at_index(
                runtime.maskfix.execute(
                    erode_dilate=0,
                    fill_holes=0,
                    remove_isolated_pixels=10,
                    smooth=1,
                    blur=0,
                    mask=mask,
                ),
                0,
            )
    return {"mode": "single_image", "frame_index": int(frame_index), "frame_image": frame_image, "mask": mask}


def run_tracking_video(
    runtime: Sam2Runtime,
    video_state: dict[str, Any],
    frame_index: int,
    object_index: int,
    positive_points: list[dict[str, float]],
    negative_points: list[dict[str, float]],
    apply_maskfix: bool = True,
) -> dict[str, Any]:
    """Track segmentation over full video from selected frame/object (mode='tracking_video')."""
    if not positive_points and not negative_points:
        raise ValueError("At least one point is required for tracking.")

    pos_coords, neg_coords = _build_points(
        runtime,
        width=video_state.get("width", 0),
        height=video_state.get("height", 0),
        positive_points=positive_points,
        negative_points=negative_points,
    )
    with torch.inference_mode():
        add_out = runtime.sam2videosegmentationaddpoints.segment(
            frame_index=int(frame_index),
            object_index=int(object_index),
            sam2_model=runtime.sam2_model,
            coordinates_positive=pos_coords,
            image=video_state["frames"],
            coordinates_negative=neg_coords,
        )
        runtime.sam2_model = get_value_at_index(add_out, 0)
        track_out = runtime.sam2videosegmentation.segment(
            keep_model_loaded=True,
            sam2_model=get_value_at_index(add_out, 0),
            inference_state=get_value_at_index(add_out, 1),
        )
        mask_sequence = get_value_at_index(track_out, 0)
        if apply_maskfix:
            mask_sequence = get_value_at_index(
                runtime.maskfix.execute(
                    erode_dilate=0,
                    fill_holes=0,
                    remove_isolated_pixels=10,
                    smooth=1,
                    blur=0,
                    mask=mask_sequence,
                ),
                0,
            )
    input_fps = _infer_input_video_fps(video_state, default_fps=24.0)
    masked_video_path = _export_masked_rgba_video(video_state, mask_sequence, fps=input_fps)
    return {
        "mode": "tracking_video",
        "frame_index": int(frame_index),
        "object_index": int(object_index),
        "fps": float(input_fps),
        "mask_sequence": mask_sequence,
        "masked_video_path": masked_video_path,
        "tracking_output": track_out,
        "sam2_model": get_value_at_index(add_out, 0),
        "inference_state": get_value_at_index(add_out, 1),
    }
