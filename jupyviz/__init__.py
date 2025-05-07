"""
Command line method to upsample a GIF with NN interpolation
-----------------------------------------------------------
convert download.gif -interpolate Nearest -filter point -resize 1000% final.gif
"""

from abc import ABC, abstractmethod
import base64
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image as PilImage
from IPython.display import Image as IpyImage, HTML as _HTML, display
from io import BytesIO
from typing import Any, List, Optional, Union

import torch
from torch import Tensor

# =============================================================================
# region Extending mediapy
# =============================================================================

import mediapy

def convert2numpy(img_pth: Tensor) -> np.ndarray:
    """
    Args:
        img_pth (Tensor, dtype=float, shape=(*BS, 3, H, W)): input
            image/images as a tensor with values in [0, 1].
    
    Returns:
        (np.ndarray, dtype=uint8, shape=(*BS, 3, H, W)): output image/images
            as a numpy array
    """
    assert img_pth.dim() >=3 and img_pth.shape[-3] == 3

    # compute new dimension permutation
    leading_dims = range(img_pth.dim() - 3)
    new_dims = (*leading_dims, -2, -1, -3)  # (*BS, 3, H, W) -> (*BS, H, W, 3)

    # based on from https://github.com/pytorch/vision/blob/3fb88b3ef1ee8107df74ca776cb57931fe3e9e1e/torchvision/utils.py#L148
    return img_pth.mul(255).add_(0.5).clamp_(0, 255).permute(*new_dims).to("cpu", torch.uint8).numpy()

def _preprocess_input(input: Any) -> np.ndarray | List:
    if isinstance(input, np.ndarray):
        return input
    if isinstance(input, Tensor):
        return convert2numpy(input)
    if isinstance(input, list):
        return [_preprocess_input(item) for item in input]
    return ValueError(f'input must be a Tensor, np.ndarray or a list of '
                      f'Tensors/np.ndarrays, but got {type(input)}')

def _squeeze_array_to_dim_3(input: np.ndarray) -> np.ndarray:
    if input.ndim == 3:
        return input
    elif input.ndim == 4:
        assert input.shape[0] == 1, 'only support batch size 1'
        return input.squeeze(0)
    else:
        raise ValueError(f'Input must have ndim=3 or 4, but got {input.ndim}')

class save (mediapy.set_show_save_dir):
    def __init__(self) -> None:
        self.temp_dir = Path('/tmp/mediapy')
        self.temp_dir.mkdir(exist_ok=True)
        print(f'Using temporary directory {self.temp_dir}')
        super().__init__(self.temp_dir)
        os.system(f'xdg-open {self.temp_dir}', )
        # subprocess.call(['xdg-open', str(self.temp_dir)])

def show_image(input, *args, **kwargs):
    input = _preprocess_input(input)
    input = _squeeze_array_to_dim_3(input)
    return mediapy.show_image(input, *args, **kwargs)

def show_images(input, *args, **kwargs):
    input = _preprocess_input(input)
    if isinstance(input, list):
        input = [_squeeze_array_to_dim_3(e) for e in input]
    return mediapy.show_images(input, *args, **kwargs)

def show_video(input, *args, **kwargs):
    input = _preprocess_input(input)
    return mediapy.show_video(input, *args, **kwargs)

def show_videos(input, *args, **kwargs):
    input = _preprocess_input(input)
    return mediapy.show_videos(input, *args, **kwargs)

# endregion
# =============================================================================
# region Custom methods
# =============================================================================

# Copied and modified from mediapy.html_from_compressed_image()
def html_from_compressed_image(
    data: bytes,
    width: int,
    height: int,
    *,
    title: str | None = None,
    border: bool | str = False,
    pixelated: bool = True,
    fmt: str = 'png',
) -> str:
  """  
  Returns an HTML string with an image tag containing encoded data.

  Args:
    data: Compressed image bytes.
    width: Width of HTML image in pixels.
    height: Height of HTML image in pixels.
    title: Optional text shown centered above image.
    border: If `bool`, whether to place a black boundary around the image, or if
      `str`, the boundary CSS style.
    pixelated: If True, sets the CSS style to 'image-rendering: pixelated;'.
    fmt: Compression encoding.
  """
  b64 = base64.b64encode(data).decode('utf-8')
  if isinstance(border, str):
    border = f'{border}; '
  elif border:
    border = 'border:1px solid black; '
  else:
    border = ''
  s_pixelated = 'pixelated' if pixelated else 'auto'
  # • Wrap <img> tag inside a <div> tag and set width and height there to force
  #   the image to be displayed at the right size (overflow if needed)
  # • Also move the 'border' CSS to the <div> tag.
  # • Set 'vertical-align: middle' in the <img> tag to avoid issues with borders
  #   - https://stackoverflow.com/a/7310398
  #   - https://www.sitepoint.com/community/t/cant-get-rid-of-space-between-images-with-padding-0-margin-0-border-0/12613
  s = (
      f'<div style="width:{width}px; height:{height}px; {border}">'
      f'  <img width="{width}" height="{height}"'
      f'   style="image-rendering:{s_pixelated}; object-fit:cover; vertical-align: middle;"'
      f'   src="data:image/{fmt};base64,{b64}"/>'
      f'</div>'
  )
  if title:
    s = f"""<div style="display:flex; align-items:left;">
      <div style="display:flex; flex-direction:column; align-items:center;">
      <div>{title}</div>{s}</div></div>"""
  return s

class HTML (_HTML):
    def display(self) -> None:
        display(self)

class _BaseHTMLElement (ABC):
    @abstractmethod 
    def html(self, **display_kwargs) -> HTML:
        raise NotImplementedError

class _BaseBioWrapper (_BaseHTMLElement):
    r"""
    For displaying images or GIFs in IPython notebooks.

    **displays** image/gif in any size we want in the notebook by modifying the
    HTML code, **stores** the image/gif in the original size.

    References:
        - https://stackoverflow.com/a/32108899
        - https://stackoverflow.com/a/72492322
    """
    @abstractmethod
    def __init__(self):
        self.bio = BytesIO()
    
    def display(self, **display_kwargs) -> IpyImage:
        """
        Display image/gif in any size we want in the notebook by modifying the
        HTML code. This does not change the size of the image/gif that is
        stored.

        Args:
            display_kwargs: keyword arguments for IpyImage, as defined in
                :func:`IPython.display.Image`

        NOTE: even though the format for GIFs should've ideally been 'gif',
        the problem is that Github does not render any other IpyImage format
        apart from 'png'.
        """       
        return IpyImage(data=self.bio.getvalue(), format='png', **display_kwargs)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save bio to file.

        Reference: https://stackoverflow.com/a/60450462
        """
        with open(path, "wb") as f:
            f.write(self.bio.getbuffer())

class img (_BaseBioWrapper):
    r"""
    For displaying images in Jupyter notebooks.

    Usage:
        jupyviz.img(img_pth).display(width=200)
    """
    def __init__(self, img: Tensor | np.ndarray | PilImage.Image) -> None:
        """
        Args:
            img ( Tensor, dtype=float, shape=(3, H, W) or (1, 3, H, W)) or
                (ndarray, dtype=uint8, shape=(H, W, 3) or (1, H, W, 3)) or
                (PIL.Image.Image, shape=(H, W))
        """
        super().__init__()

        if isinstance(img, (Tensor, np.ndarray)):
            img: np.ndarray = _preprocess_input(img)
            img: np.ndarray = _squeeze_array_to_dim_3(img)
            img_pil = PilImage.fromarray(img)
        elif isinstance(img, PilImage.Image):
            img_pil = img
        else:
            raise ValueError(f'img must be a Tensor, np.ndarray or a PIL Image'
                             f', but got {type(img)}')

        # save image into buffer
        img_pil.save(self.bio, format='png')
 
    def html(self, width=None, height=None, title=None, border=False, pixelated=True) -> HTML:
        html_str = html_from_compressed_image(
            data=self.bio.getvalue(),
            width=width,
            height=height,
            title=title,
            border=border,
            pixelated=pixelated,
            fmt='png',
        )
        return HTML(html_str)

class gif (_BaseBioWrapper):
    r"""
    For displaying GIFs in Jupyter notebooks.

    Online GIF analysis tool: https://onlinegiftools.com/analyze-gif

    Usage:
        jupyviz.gif(imgs_pth).display(width=200)
    """
    def __init__(self,
                 imgs: Union[Tensor, np.ndarray, List[Tensor | np.ndarray]],
                 time_in_ms: int=2000,
                 hold_last_frame_time_in_ms: int=0,
                 loop: Optional[int]=0) -> None:
        r"""
        Args:
            imgs ( Tensor, dtype=float, shape=(B, 3, H, W)) or
                 (ndarray, dtype=uint8, shape=(B, H, W, 3)) or
                 (List[( Tensor, dtype=float, shape=(3, H, W)) or (1, 3, H, W))]) or
                 (List[(ndarray, dtype=uint8, shape=(H, W, 3)) or (1, H, W, 3))]) or
            time_in_ms (int): total time for the GIF to play in milliseconds
            hold_last_frame_time_in_ms (int): extra time to hold the last frame
            loop (int): number of times to loop the GIF
                - loop=0     # loop endlessly
                - loop=k     # loop k+1 times
                - loop=None  # loop only once

        References:
          - https://stackoverflow.com/a/51075068
        """
        super().__init__()

        imgs_np: np.ndarray | List[np.ndarray] = _preprocess_input(imgs)
        if isinstance(imgs_np, np.ndarray):
            assert imgs.ndim == 4
            imgs_np: List[np.ndarray] = list(imgs)
        imgs_np = [_squeeze_array_to_dim_3(img) for img in imgs_np]  
        imgs_pil = [PilImage.fromarray(img_np) for img_np in imgs_np]

        # a minimum duration of 10ms per frame is required for imageio/PIL.
        # in fact, frame durations are in increments of 10ms only!
        duration_in_ms = max(time_in_ms // len(imgs_pil), 10)
        durations_in_ms = [duration_in_ms for _ in imgs_pil]

        # hold last frame for extra duration
        durations_in_ms[-1] += hold_last_frame_time_in_ms

        # save GIF into buffer
        # explanation of parameters: https://pillow.readthedocs.io/en/latest/handbook/image-file-formats.html#gif-saving
        imgs_pil[0].save(self.bio,
                         format='gif',
                         save_all=True,
                         append_images=imgs_pil[1:],
                         duration=durations_in_ms,
                         loop=loop)

    def html(self, width=None, height=None, title=None, border=False, pixelated=True) -> HTML:
        """
        References:
            - https://stackoverflow.com/a/32108899
            - https://github.com/ipython/ipython/issues/10045#issuecomment-642640541
        """
        html_str = html_from_compressed_image(
            data=self.bio.getvalue(),
            width=width,
            height=height,
            title=title,
            border=border,
            pixelated=pixelated,
            fmt='gif',
        )
        return HTML(html_str)

class tile (_BaseHTMLElement):
    r"""
    For tiling images or GIFs in Jupyter notebooks.
    Based off of mediapy.show_images() and mediapy.show_videos()
    """
    def __init__(self, elements: List[_BaseHTMLElement]) -> None:
        """
        Args:
            elements (List[_BaseHTMLElement]): A list of elements to tile.
        """
        super().__init__()

        self.elements = elements

    def html(self,
             width=None, height=None,
             border=False,
             pixelated=True,
             rows: Optional[List[int]]=None,
             ylabel: Optional[str]=None,
             ylabels: Optional[List[str]]=None,
             titles: Optional[List[str]]=None) -> HTML:
             
        """
        Args:
            width (int): width of each element
            height (int): height of each element
            pixelated (bool): whether to pixelate each element
            rows (List[int]): number of elements in each row
            ylabel (str, optional): ylabel (only if a single row)
            ylabels (List[str], optional): ylabel for each row (s)
            titles (List[str], optional): title for each element
        """
        # set default rows
        if rows is None:
            rows = [len(self.elements)]

        # check if rows are valid
        assert sum(rows) == len(self.elements), \
            f'rows={rows} must have a sum of {len(self.elements)}'

        # set default ylabel:
        assert (ylabel is None) or (ylabels is None)
        if ylabel is not None:
            ylabels = [ylabel]
        
        # check if ylabel are valid
        if ylabels is not None:
            assert len(ylabels) == len(rows), \
                f'The number of ylabels must equal the number of rows={len(rows)}'

        # set default titles
        if titles is None:
            titles = [None for _ in self.elements]
        
        # check if title are valid
        assert len(titles) == len(self.elements), \
            f'The number of titles must equal the number of elements={len(self.elements)}'

        # get HTML for each element
        html_strings: List[str] = []
        for i in range(len(self.elements)):
            element, title = self.elements[i], titles[i]
            html = element.html(width=width,
                                height=height,
                                title=title,
                                border=border,
                                pixelated=pixelated)
            html_strings.append(html.data)
        
        # Create single-row tables each with no more than 'columns' elements.
        table_strings: List[str] = []
        index = 0
        for i, num_cols in enumerate(rows):
            row_html_strings = html_strings[index: index + num_cols]
            index += num_cols

            td = '<td style="padding:1px;">'
            s = ''.join(f'{td}{e}</td>' for e in row_html_strings)
            if ylabels is not None:
                ylabel = ylabels[i]
                style = 'writing-mode:vertical-lr; transform:rotate(180deg);'
                s = f'{td}<span style="{style}">{ylabel}</span></td>' + s
            table_strings.append(
                f'<table'
                f' style="border-spacing:0px;"><tr>{s}</tr></table>'
            )
        return HTML(''.join(table_strings))

# endregion
# =============================================================================
# region Interactive viewer
# =============================================================================
    
class iview:
    def __init__(self, active: bool=True, window_title: str=''):
        self.active = active
        if not self.active: return

        plt.ion()

        self.fig = plt.figure(num=window_title)
        self.imshow = None

        plt.axis('off')
        self.fig.tight_layout()
    
    def __enter__(self) -> 'iview':
        return self
    
    def update(self, img: Tensor | np.ndarray) -> None:
        if not self.active: return
        if not plt.fignum_exists(self.fig.number):
            self.close()
            return

        img = _preprocess_input(img)
        img = _squeeze_array_to_dim_3(img)

        if self.imshow is None:
            self.imshow = plt.imshow(img)
        else:
            self.imshow.set_data(img)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self) -> None:
        if not self.active: return
        plt.close(self.fig)
    
    def __exit__(self, type, value, traceback) -> None:
        self.close()

# endregion
# =============================================================================
# Utility functions
# =============================================================================

def GetNumpyImageFromMatplotlibAxis(ax: plt.Axes, dpi, pad_inches: float=0.0) -> np.ndarray:
    buf = BytesIO()
    ax.figure.savefig(buf, format='png', bbox_inches='tight', dpi=dpi,pad_inches=pad_inches)
    buf.seek(0)
    image = np.array(PilImage.open(buf))  # This gives uint8 directly
    buf.close()
    return image

# endregion
# =============================================================================
