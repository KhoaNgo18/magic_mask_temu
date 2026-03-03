import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes

    sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS

_CUSTOM_NODES_IMPORTED = False


def _extract_video_path(obj: Any) -> str | None:
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
            found = _extract_video_path(value)
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _extract_video_path(item)
            if found:
                return found
        return None
    return None


def ensure_custom_nodes_imported() -> None:
    global _CUSTOM_NODES_IMPORTED
    if _CUSTOM_NODES_IMPORTED:
        return
    import_custom_nodes()
    _CUSTOM_NODES_IMPORTED = True


def interpolate_video(
    input_video_path: str,
    output_prefix: str | None = None,
    ckpt_name: str = "rife49.pth",
    multiplier: int = 2,
    fast_mode: bool = False,
    ensemble: bool = True,
    scale_factor: int = 1,
    clear_cache_after_n_frames: int = 10,
    crf: int = 19,
) -> dict[str, Any]:
    if not input_video_path:
        raise ValueError("input_video_path is required.")
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    ensure_custom_nodes_imported()

    filename_prefix = output_prefix or f"interpolated_{os.path.splitext(os.path.basename(input_video_path))[0]}"

    with torch.inference_mode():
        vhs_loadvideoffmpeg = NODE_CLASS_MAPPINGS["VHS_LoadVideoFFmpeg"]()
        rife_vfi = NODE_CLASS_MAPPINGS["RIFE VFI"]()
        vhs_videoinfo = NODE_CLASS_MAPPINGS["VHS_VideoInfo"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        loaded = vhs_loadvideoffmpeg.load_video(
            video=input_video_path,
            force_rate=0,
            custom_width=0,
            custom_height=0,
            frame_load_cap=0,
            start_time=0,
            format="AnimateDiff",
            unique_id=random.randint(1, 2**63 - 1),
        )

        interpolated = rife_vfi.vfi(
            ckpt_name=ckpt_name,
            clear_cache_after_n_frames=clear_cache_after_n_frames,
            multiplier=multiplier,
            fast_mode=fast_mode,
            ensemble=ensemble,
            scale_factor=scale_factor,
            frames=get_value_at_index(loaded, 0),
        )

        video_info = vhs_videoinfo.get_video_info(video_info=get_value_at_index(loaded, 3))

        combined = vhs_videocombine.combine_video(
            frame_rate=get_value_at_index(video_info, 5),
            loop_count=0,
            filename_prefix=filename_prefix,
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=crf,
            save_metadata=True,
            trim_to_audio=False,
            pingpong=False,
            save_output=True,
            images=get_value_at_index(interpolated, 0),
            unique_id=random.randint(1, 2**63 - 1),
        )

    output_video_path = _extract_video_path(combined)
    return {
        "filename_prefix": filename_prefix,
        "video_path": output_video_path,
        "raw_output": combined,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python frame_interpolate.py <input_video_path>")
        return
    result = interpolate_video(input_video_path=sys.argv[1])
    print(f"Output video: {result.get('video_path')}")


if __name__ == "__main__":
    main()
