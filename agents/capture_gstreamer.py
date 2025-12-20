import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def build_gstreamer_pipeline(
    device: str,
    width: int,
    height: int,
    fps: float,
    use_nvmm: bool,
) -> str:
    override = os.getenv("GST_PIPELINE", "").strip()
    if override:
        return override

    source = os.getenv("GST_SOURCE", "v4l2src").strip()
    if source == "v4l2src":
        source = f"v4l2src device={device}"
    elif source == "nvarguscamerasrc":
        sensor_id = _int_env("GST_SENSOR_ID", 0)
        source = f"nvarguscamerasrc sensor-id={sensor_id}"

    caps = ["video/x-raw"]
    if width > 0:
        caps.append(f"width={width}")
    if height > 0:
        caps.append(f"height={height}")
    if fps > 0:
        caps.append(f"framerate={int(fps)}/1")
    caps_str = ",".join(caps)

    drop = _int_env("GST_DROP", 1)
    max_buffers = _int_env("GST_MAX_BUFFERS", 1)
    sync = os.getenv("GST_SYNC", "0").strip()

    if use_nvmm:
        return (
            f"{source} ! {caps_str} ! nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop={drop} max-buffers={max_buffers} sync={sync}"
        )
    return (
        f"{source} ! {caps_str} ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop={drop} max-buffers={max_buffers} sync={sync}"
    )
