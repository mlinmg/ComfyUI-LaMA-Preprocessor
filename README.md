# LaMa Preprocessor (WIP)
The best results are given on landscape inpaint, not so much in animation.

This preprocessor finally enable user to generate coherent inpaint and outpaint prompt-free
## A LaMa prerocessor for ComfyUi
This is a simple workflow example. You can use it downloading workflows/workflow_lama.json and then dropping it a comfyui tab
![workflow](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/73d422ae-a30c-43de-acf1-44bbfaa28530)

---

This are some non cherry picked examples all based on this image
![startingImage](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/6d6067fd-dfa9-414b-99c8-3151d0a15dc7)

---

![ComfyUI_01581_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/58657498-e57a-49e5-910b-d7ee309ca26c)
![ComfyUI_01580_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/1131312a-d77a-496d-960d-54450eb1dde0)
![ComfyUI_01579_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/a2d2a1a9-d2a8-4533-9227-678b9dd7f313)
![ComfyUI_01578_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/5da44552-d3ab-468c-bc11-a1e5e8181b75)


You can find the processor in image/preprocessors
## Contributing

Everyone is invited to contribute, to do so you can just make a pull request

If you would like to help to the development of this repo there are some missing features that still need to be implemented:
- [ ] This workflow exibits some image darkening/color shifting, this should be further investigated in order to fix it
- [ ] More consistent results. One of the problem might be in [this function](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/blob/main/inpaint_Lama.py#L179) it seems that sometimes the image does not match the mask and if you pass this image to the LaMa model it make a noisy greyish mess
- [ ] [soft injection](https://github.com/Mikubill/sd-webui-controlnet/blob/7a4805c8ea3256a0eab3512280bd4f84ca0c8182/scripts/hook.py#L620), since I cannot understand how to set different weights inside the sampling steps in comfyUI. At the moment the control seems too high all around, and this cause artifacts in more complex in/outpaint
- [x] LaMa preprocessing node
- [x] Basic workflow

## Thanks

This code is inspired by Mikubill/sd-webui-controlnet
