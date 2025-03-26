from typing import List
import numpy as np
import jupyviz as jviz


def show_gif(imgs: List[np.ndarray], fps: float = 25.0, width: int = 256) -> None:
    duration_in_ms = int(len(imgs) / fps * 1000)
    gif = jviz.gif(
        imgs,
        time_in_ms=duration_in_ms,
        hold_last_frame_time_in_ms=2000
    )
    gif.html(width=width, pixelated=False).display()
