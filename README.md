# Fork of [BlueAmulet's fork](https://github.com/BlueAmulet/ESRGAN) of [ESRGAN by Xinntao](https://github.com/xinntao/ESRGAN)

This fork ports features over from my ESRGAN-Bot repository. It natively allows:
* Splitting/Merging functionality
* Seamless texture preservation
* Model chaining
* Transparency preservation

To change the tile size for the split/merge functionality, use the `--tile_size` argument

To set your textures to seamless, use the `--seamless` flag

To chain models, simply put one model name after another with a `>` in between, such as `1xDeJpeg.pth>4xESRGAN.pth` **note: model names must be the complete full name, and the models must be present in your `/models` folder. Unlike normal ESRGAN, you should not include `/models` in the model names.

To use 1 bit binary alpha transparency, set the `--binary_alpha` flag to True. When using `--binary_alpha` transparency, provide the optional `--alpha_threshold` to specify the alpha transparency threshold. 1 bit binary transparency is useful when upscaling images that require that the end result has 1 bit transparency, e.g. PSX games.

Examples: 
* `python test.py 4xBox.pth --seamless`
* `python test.py 1xSSAntiAlias9x.pth>4xBox.pth --tile_size=800`
* `python test.py 4xBox.pth --binary_alpha True --alpha_threshold .2`
