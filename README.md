# Fork of [BlueAmulet's fork](https://github.com/BlueAmulet/ESRGAN) of [ESRGAN by Xinntao](https://github.com/xinntao/ESRGAN)

This fork ports features over from my ESRGAN-Bot repository. It natively allows:
* Splitting/Merging functionality
* Seamless texture preservation
* Model chaining
* Transparency preservation

To change the tile size for the split/merge functionality, use the `--tile_size` argument

To set your textures to seamless, use the `--seamless` flag

To chain models, simply put one model name after another with a `>` in between, such as `1xDeJpeg.pth>4xESRGAN.pth` **note: model names must be the complete full name, and the models must be present in your `/models` folder. Unlike normal ESRGAN, you should not include `/models` in the model names.
