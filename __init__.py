import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from mhg.upscale.ESRGAN.upscale import Upscale, SeamlessOptions, AlphaOptions
from rich.logging import RichHandler


from mhg.config import config


@dataclass
class UpscaleApp:
    model: str
    input: Path
    output: Path
    reverse: bool = False
    skip_existing: bool = False
    delete_input: bool = False
    seamless: Optional[SeamlessOptions] = None
    cpu: bool = False
    fp16: bool = False
    device_id: int = 0
    cache_max_split_depth: bool = False
    binary_alpha: bool = False
    ternary_alpha: bool = False
    alpha_threshold: float = 0.5
    alpha_boundary_offset: float = 0.2
    alpha_mode: Optional[AlphaOptions] = None
    verbose: bool = False

    def process(self):
        upscale = Upscale(
            model=self.model,
            input=self.input,
            output=self.output,
            reverse=self.reverse,
            skip_existing=self.skip_existing,
            delete_input=self.delete_input,
            seamless=self.seamless,
            cpu=self.cpu,
            fp16=self.fp16,
            device_id=self.device_id,
            cache_max_split_depth=self.cache_max_split_depth,
            binary_alpha=self.binary_alpha,
            ternary_alpha=self.ternary_alpha,
            alpha_threshold=self.alpha_threshold,
            alpha_boundary_offset=self.alpha_boundary_offset,
            alpha_mode=self.alpha_mode,
        )
        upscale.run()
