import json
import os
from typing import Any

import gradio as gr
import numpy as np
import torch

from sam_2 import (
    Sam2Runtime,
    get_frame_image,
    init_runtime,
    load_video_state,
    run_single_image,
    run_tracking_video,
)


def _to_numpy_image(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) if np.issubdtype(arr.dtype, np.floating) else arr
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _to_numpy_mask(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) if np.issubdtype(arr.dtype, np.floating) else arr
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _draw_points(
    image: np.ndarray,
    pos_points: list[dict[str, float]],
    neg_points: list[dict[str, float]],
    selected_point: dict[str, Any] | None = None,
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    selected_type = selected_point.get("type") if isinstance(selected_point, dict) else None
    selected_index = selected_point.get("index") if isinstance(selected_point, dict) else None

    for idx, point in enumerate(pos_points):
        x = int(round(point["x"]))
        y = int(round(point["y"]))
        if 0 <= x < w and 0 <= y < h:
            is_selected = selected_type == "positive" and selected_index == idx
            size = 7 if is_selected else 3
            color = [255, 255, 0] if is_selected else [0, 255, 0]
            out[max(0, y - size) : min(h, y + size + 1), max(0, x - size) : min(w, x + size + 1)] = color
    for idx, point in enumerate(neg_points):
        x = int(round(point["x"]))
        y = int(round(point["y"]))
        if 0 <= x < w and 0 <= y < h:
            is_selected = selected_type == "negative" and selected_index == idx
            size = 7 if is_selected else 3
            color = [255, 255, 0] if is_selected else [255, 0, 0]
            out[max(0, y - size) : min(h, y + size + 1), max(0, x - size) : min(w, x + size + 1)] = color
    return out


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        return image
    out = image.copy().astype(np.float32)
    active = mask > 0
    out[active, 0] = np.minimum(255.0, out[active, 0] * 0.5 + 120.0)
    out[active, 1] = out[active, 1] * 0.6
    out[active, 2] = out[active, 2] * 0.6
    return out.astype(np.uint8)


def _points_json(pos_points: list[dict[str, float]], neg_points: list[dict[str, float]]) -> str:
    return json.dumps({"positive": pos_points, "negative": neg_points}, indent=2)


def _points_index_text(pos_points: list[dict[str, float]], neg_points: list[dict[str, float]]) -> str:
    pos_lines = [
        f"[{idx}] x={point['x']:.1f}, y={point['y']:.1f}" for idx, point in enumerate(pos_points)
    ]
    neg_lines = [
        f"[{idx}] x={point['x']:.1f}, y={point['y']:.1f}" for idx, point in enumerate(neg_points)
    ]
    pos_block = "\n".join(pos_lines) if pos_lines else "(empty)"
    neg_block = "\n".join(neg_lines) if neg_lines else "(empty)"
    return f"positive points:\n{pos_block}\n\nnegative points:\n{neg_block}"


def _points_table_rows(pos_points: list[dict[str, float]], neg_points: list[dict[str, float]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, point in enumerate(pos_points):
        rows.append(["positive", idx, round(float(point["x"]), 2), round(float(point["y"]), 2), "Delete"])
    for idx, point in enumerate(neg_points):
        rows.append(["negative", idx, round(float(point["x"]), 2), round(float(point["y"]), 2), "Delete"])
    return rows


def _selected_from_row(row_idx: int, pos_points: list[dict[str, float]], neg_points: list[dict[str, float]]) -> dict[str, Any] | None:
    if row_idx < 0:
        return None
    pos_count = len(pos_points)
    if row_idx < pos_count:
        return {"type": "positive", "index": row_idx}
    neg_idx = row_idx - pos_count
    if neg_idx < len(neg_points):
        return {"type": "negative", "index": neg_idx}
    return None


def _single_view_image(frame_with_points: np.ndarray | None, overlay: np.ndarray | None) -> np.ndarray | None:
    return overlay if overlay is not None else frame_with_points


def _find_video_path(obj: Any) -> str | None:
    def _comfy_media_candidates(filename: str, subfolder: str, media_type: str) -> list[str]:
        candidates: list[str] = []
        if os.path.isabs(filename):
            candidates.append(filename)
            return candidates

        try:
            import folder_paths  # type: ignore

            roots = {
                "output": folder_paths.get_output_directory(),
                "temp": folder_paths.get_temp_directory(),
                "input": folder_paths.get_input_directory(),
            }
            preferred_root = roots.get(media_type, roots["output"])
            candidates.append(os.path.join(preferred_root, subfolder, filename))
            for root in roots.values():
                candidates.append(os.path.join(root, subfolder, filename))
                candidates.append(os.path.join(root, filename))
        except Exception:
            pass

        candidates.append(os.path.abspath(filename))
        if subfolder:
            candidates.append(os.path.abspath(os.path.join(subfolder, filename)))
        return candidates

    if isinstance(obj, str):
        lower = obj.lower()
        if lower.endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
            if os.path.exists(obj):
                return obj
            abs_obj = os.path.abspath(obj)
            if os.path.exists(abs_obj):
                return abs_obj
        return None
    if isinstance(obj, dict):
        filename = obj.get("filename")
        if isinstance(filename, str) and filename.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi")):
            subfolder = str(obj.get("subfolder") or "")
            media_type = str(obj.get("type") or "output")
            for candidate in _comfy_media_candidates(filename, subfolder, media_type):
                if os.path.exists(candidate):
                    return candidate
        for value in obj.values():
            found = _find_video_path(value)
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_video_path(item)
            if found:
                return found
        return None
    return None


def _infer_frame_count(video_state: dict[str, Any] | None) -> int:
    if not video_state:
        return 1
    count = video_state.get("frame_count")
    if isinstance(count, (int, float)) and int(count) > 0:
        return int(count)
    frames = video_state.get("frames")
    try:
        return max(1, int(len(frames)))
    except Exception:
        pass
    if isinstance(frames, torch.Tensor) and frames.ndim > 0:
        return max(1, int(frames.shape[0]))
    return 1


def _extract_uploaded_video_path(video_input: Any) -> str | None:
    if isinstance(video_input, str):
        return video_input or None
    if isinstance(video_input, dict):
        for key in ("path", "video", "name", "filename"):
            value = video_input.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _clamp_frame_index(frame_index: int | float | None, video_state: dict[str, Any] | None) -> int:
    frame_count = _infer_frame_count(video_state)
    idx = int(frame_index or 0)
    return max(0, min(idx, frame_count - 1))


def _extract_click_xy(evt: gr.SelectData) -> tuple[float, float] | None:
    if evt is None:
        return None

    index = getattr(evt, "index", None)
    if isinstance(index, (tuple, list)) and len(index) >= 2 and index[0] is not None and index[1] is not None:
        return float(index[0]), float(index[1])
    if isinstance(index, dict) and index.get("x") is not None and index.get("y") is not None:
        return float(index["x"]), float(index["y"])

    value = getattr(evt, "value", None)
    if isinstance(value, dict) and value.get("x") is not None and value.get("y") is not None:
        return float(value["x"]), float(value["y"])
    return None


def _render_frame_with_current_points(
    runtime: Sam2Runtime | None,
    video_state: dict[str, Any] | None,
    frame_index: int,
    pos_points: list[dict[str, float]],
    neg_points: list[dict[str, float]],
    selected_point: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if runtime is None or video_state is None:
        return None, None, "Upload a video first."
    frame = _to_numpy_image(get_frame_image(runtime, video_state, int(frame_index)))
    frame_with_points = _draw_points(frame, pos_points, neg_points, selected_point=selected_point)
    if not pos_points and not neg_points:
        return frame_with_points, None, "Add at least one point to run single_image segmentation."
    result = run_single_image(
        runtime=runtime,
        video_state=video_state,
        frame_index=int(frame_index),
        positive_points=pos_points,
        negative_points=neg_points,
        apply_maskfix=True,
    )
    mask = _to_numpy_mask(result["mask"])
    overlay = _overlay_mask(frame_with_points, mask)
    return frame_with_points, overlay, "single_image segmentation updated."


def _init_or_reuse_runtime(runtime_state: Sam2Runtime | None) -> Sam2Runtime:
    if runtime_state is not None:
        return runtime_state
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if device == "cuda" else "fp32"
    return init_runtime(device=device, precision=precision)


def on_upload_video(video_input: Any, runtime_state: Sam2Runtime | None):
    try:
        video_path = _extract_uploaded_video_path(video_input)
        if not video_path:
            raise ValueError("No uploaded video path found.")
        runtime = _init_or_reuse_runtime(runtime_state)
        video_state = load_video_state(runtime, video_path)
        frame_count = _infer_frame_count(video_state)
        frame_index = 0
        pos_points: list[dict[str, float]] = []
        neg_points: list[dict[str, float]] = []
        selected_point = None
        frame = _to_numpy_image(get_frame_image(runtime, video_state, frame_index))
        frame = _draw_points(frame, pos_points, neg_points, selected_point=selected_point)
        slider_update = gr.update(minimum=0, maximum=frame_count - 1, value=0, step=1)
        msg = f"Video loaded. Frames: {frame_count}. Add positive/negative points."
        return (
            runtime,
            video_state,
            frame_index,
            pos_points,
            neg_points,
            selected_point,
            slider_update,
            _single_view_image(frame, None),
            _points_table_rows(pos_points, neg_points),
            msg,
        )
    except Exception as exc:
        return (
            runtime_state,
            None,
            0,
            [],
            [],
            None,
            gr.update(),
            None,
            _points_table_rows([], []),
            f"Load failed: {exc}",
        )


def on_frame_slider_change(frame_index: int, runtime, video_state, pos_points, neg_points, selected_point):
    try:
        current_frame = _clamp_frame_index(frame_index, video_state)
        frame, overlay, msg = _render_frame_with_current_points(
            runtime,
            video_state,
            current_frame,
            pos_points,
            neg_points,
            selected_point=selected_point,
        )
        return current_frame, _single_view_image(frame, overlay), _points_table_rows(pos_points, neg_points), msg
    except Exception as exc:
        current_frame = _clamp_frame_index(frame_index, video_state)
        return current_frame, None, _points_table_rows(pos_points, neg_points), f"Frame update failed: {exc}"


def on_add_point(point_mode: str, frame_index: int, runtime, video_state, pos_points, neg_points, evt: gr.SelectData):
    try:
        click_xy = _extract_click_xy(evt)
        if click_xy is None:
            return (
                pos_points,
                neg_points,
                None,
                gr.update(),
                _points_table_rows(pos_points, neg_points),
                "Error while adding point(s): invalid click position.",
            )
        x, y = click_xy
        point = {"x": x, "y": y}
        if point_mode == "positive":
            pos_points = [*pos_points, point]
        else:
            neg_points = [*neg_points, point]
        selected_point = None
        current_frame = _clamp_frame_index(frame_index, video_state)
        frame, overlay, msg = _render_frame_with_current_points(
            runtime,
            video_state,
            current_frame,
            pos_points,
            neg_points,
            selected_point=selected_point,
        )
        return (
            pos_points,
            neg_points,
            selected_point,
            _single_view_image(frame, overlay),
            _points_table_rows(pos_points, neg_points),
            msg,
        )
    except Exception as exc:
        return (
            pos_points,
            neg_points,
            None,
            None,
            _points_table_rows(pos_points, neg_points),
            f"Add point failed: {exc}",
        )


def on_points_table_select(
    frame_index: int,
    runtime,
    video_state,
    pos_points,
    neg_points,
    evt: gr.SelectData,
):
    try:
        if evt is None or evt.index is None or not isinstance(evt.index, (tuple, list)) or len(evt.index) < 2:
            return (
                pos_points,
                neg_points,
                None,
                gr.update(),
                _points_table_rows(pos_points, neg_points),
                "Delete point failed: invalid table selection.",
            )
        row_idx, col_idx = int(evt.index[0]), int(evt.index[1])
        selected_point = _selected_from_row(row_idx, pos_points, neg_points)
        if selected_point is None:
            return (
                pos_points,
                neg_points,
                None,
                gr.update(),
                _points_table_rows(pos_points, neg_points),
                "Selection failed: row index out of range.",
            )

        if col_idx != 4:
            current_frame = _clamp_frame_index(frame_index, video_state)
            frame, overlay, _ = _render_frame_with_current_points(
                runtime,
                video_state,
                current_frame,
                pos_points,
                neg_points,
                selected_point=selected_point,
            )
            return (
                pos_points,
                neg_points,
                selected_point,
                _single_view_image(frame, overlay),
                _points_table_rows(pos_points, neg_points),
                f"Selected point: {selected_point['type']}[{selected_point['index']}].",
            )

        pos_count = len(pos_points)
        if row_idx < 0 or row_idx >= (pos_count + len(neg_points)):
            return (
                pos_points,
                neg_points,
                None,
                gr.update(),
                _points_table_rows(pos_points, neg_points),
                "Delete point failed: row index out of range.",
            )

        if row_idx < pos_count:
            pos_points = [*pos_points[:row_idx], *pos_points[row_idx + 1 :]]
        else:
            neg_idx = row_idx - pos_count
            neg_points = [*neg_points[:neg_idx], *neg_points[neg_idx + 1 :]]

        selected_point = None
        current_frame = _clamp_frame_index(frame_index, video_state)
        frame, overlay, msg = _render_frame_with_current_points(
            runtime,
            video_state,
            current_frame,
            pos_points,
            neg_points,
            selected_point=selected_point,
        )
        return (
            pos_points,
            neg_points,
            selected_point,
            _single_view_image(frame, overlay),
            _points_table_rows(pos_points, neg_points),
            msg,
        )
    except Exception as exc:
        return (
            pos_points,
            neg_points,
            None,
            None,
            _points_table_rows(pos_points, neg_points),
            f"Delete point failed: {exc}",
        )


def on_clear_points(frame_index: int, runtime, video_state):
    pos_points: list[dict[str, float]] = []
    neg_points: list[dict[str, float]] = []
    selected_point = None
    try:
        current_frame = _clamp_frame_index(frame_index, video_state)
        frame, overlay, msg = _render_frame_with_current_points(
            runtime,
            video_state,
            current_frame,
            pos_points,
            neg_points,
            selected_point=selected_point,
        )
        return (
            pos_points,
            neg_points,
            selected_point,
            _single_view_image(frame, overlay),
            _points_table_rows(pos_points, neg_points),
            msg,
        )
    except Exception as exc:
        return (
            pos_points,
            neg_points,
            selected_point,
            None,
            _points_table_rows(pos_points, neg_points),
            f"Clear points failed: {exc}",
        )


def on_track(frame_index: int, runtime, video_state, pos_points, neg_points):
    try:
        current_frame = _clamp_frame_index(frame_index, video_state)
        current_object = current_frame
        result = run_tracking_video(
            runtime=runtime,
            video_state=video_state,
            frame_index=current_frame,
            object_index=current_object,
            positive_points=pos_points,
            negative_points=neg_points,
            apply_maskfix=True,
        )
        masked_video_path = result.get("masked_video_path")
        if isinstance(masked_video_path, str) and os.path.exists(masked_video_path):
            msg = f"tracking_video completed from frame_index={current_frame}. Masked RGBA video displayed."
            return masked_video_path, msg
        tracking_video_path = _find_video_path(result.get("tracking_output"))
        if tracking_video_path:
            msg = f"tracking_video completed from frame_index={current_frame}. Video displayed."
            return tracking_video_path, msg
        msg = (
            f"tracking_video completed from frame_index={current_frame}, object_index={current_object}, "
            "but no video path was found in node output."
        )
        return None, msg
    except Exception as exc:
        return None, f"Tracking failed: {exc}"


def build_app(initial_runtime: Sam2Runtime | None = None, startup_status: str = "Upload a video to start.") -> gr.Blocks:
    with gr.Blocks(title="SAM2 Video Segmentation") as demo:
        gr.Markdown("## SAM2 Video Segmentation (single_image + tracking_video)")

        runtime_state = gr.State(initial_runtime)
        video_state = gr.State(None)
        current_frame_state = gr.State(0)
        positive_points_state = gr.State([])
        negative_points_state = gr.State([])
        selected_point_state = gr.State(None)

        with gr.Row():
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            with gr.Column():
                frame_slider = gr.Slider(label="Frame Index", minimum=0, maximum=0, value=0, step=1)
                point_mode = gr.Radio(choices=["positive", "negative"], value="positive", label="Point Type")
                clear_points_btn = gr.Button("Clear All Points")
                track_btn = gr.Button("Start Tracking")

        with gr.Row():
            frame_view = gr.Image(
                label="single_image (click to add point)",
                type="numpy",
                interactive=True,
                height=720,
            )

        with gr.Row():
            tracking_view = gr.Video(label="tracking_video Output")
            points_view = gr.Dataframe(
                headers=["type", "index", "x", "y", "delete"],
                datatype=["str", "number", "number", "number", "str"],
                value=_points_table_rows([], []),
                interactive=False,
                wrap=True,
                label="Points Table (click Delete cell to remove point)",
            )

        status = gr.Textbox(label="Status", value=startup_status, interactive=False)

        video_input.change(
            fn=on_upload_video,
            inputs=[video_input, runtime_state],
            outputs=[
                runtime_state,
                video_state,
                current_frame_state,
                positive_points_state,
                negative_points_state,
                selected_point_state,
                frame_slider,
                frame_view,
                points_view,
                status,
            ],
        )

        frame_slider.change(
            fn=on_frame_slider_change,
            inputs=[frame_slider, runtime_state, video_state, positive_points_state, negative_points_state, selected_point_state],
            outputs=[current_frame_state, frame_view, points_view, status],
        )

        frame_view.select(
            fn=on_add_point,
            inputs=[point_mode, current_frame_state, runtime_state, video_state, positive_points_state, negative_points_state],
            outputs=[positive_points_state, negative_points_state, selected_point_state, frame_view, points_view, status],
        )

        points_view.select(
            fn=on_points_table_select,
            inputs=[current_frame_state, runtime_state, video_state, positive_points_state, negative_points_state],
            outputs=[positive_points_state, negative_points_state, selected_point_state, frame_view, points_view, status],
        )

        clear_points_btn.click(
            fn=on_clear_points,
            inputs=[current_frame_state, runtime_state, video_state],
            outputs=[positive_points_state, negative_points_state, selected_point_state, frame_view, points_view, status],
        )

        track_btn.click(
            fn=on_track,
            inputs=[current_frame_state, runtime_state, video_state, positive_points_state, negative_points_state],
            outputs=[tracking_view, status],
        )

    return demo


if __name__ == "__main__":
    preloaded_runtime = None
    preload_msg = "Upload a video to start."
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = "fp16" if device == "cuda" else "fp32"
        preloaded_runtime = init_runtime(device=device, precision=precision)
        preload_msg = f"SAM2/Comfy runtime ready on {device}. Upload a video to start."
    except Exception as exc:
        preload_msg = f"Startup warning: could not preload runtime ({exc}). Upload will retry initialization."

    app = build_app(initial_runtime=preloaded_runtime, startup_status=preload_msg)
    app.launch()
