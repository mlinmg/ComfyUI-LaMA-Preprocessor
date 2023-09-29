# LaMa Preprocessor (WIP)

This preprocessor finally enable users to generate coherent inpaint and outpaint **prompt-free**

For inpainting tasks, it's recommended to use the 'outpaint' function. Although the 'inpaint' function is still in the development phase, the results from the 'outpaint' function remain quite satisfactory.

The best results are given on landscapes, good results can still be achieved in drawings by lowering the controlnet end percentage to 0.7-0.8

## Installation

1) Manager installation: be sure to have [ComfyUi Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, then just search for lama preprocessor  

2) Manual Installation: clone this repo inside the custom_nodes folder
## A LaMa prerocessor for ComfyUi
This is a simple workflow example.To use this, download workflows/workflow_lama.json and then drop it in a ComfyUI tab
![workflow](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/098f40d6-307c-4ad7-b0c0-5f40ea2e777f)

---


This are some non cherry picked results, all obtained starting from this image

![startingImage](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/18b937d6-bcda-4606-a3b0-b24af55d27dd)


---

![ComfyUI_01581_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/3adbc1f8-bb3e-4ae5-b31b-d7fb8624f0ae)
![ComfyUI_01580_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/77f73d96-2612-431d-bd3c-7e6f4c2503c0)
![ComfyUI_01579_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/5715229b-6b6e-4f2e-917c-97b09758c805)
![ComfyUI_01578_](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/assets/121761685/dbe1a705-7574-4b2e-a2c6-d06708a38261)


You can find the processor in image/preprocessors
## Contributing

Everyone is invited to contribute, to do so you can just make a pull request

If you would like to help to the development of this repo there are some missing features that still need to be implemented:
- [ ] An unusual behavior is observed when providing an empty prompt to the drawing/cartoon outpainting system. Interestingly, the results appear to be significantly better when the two conditionings are prompted with "positive" and "negative" respectively. This warrants further investigation.
- [ ] This workflow exibits some image darkening/color shifting, this should be further investigated in order to fix it
- [ ] More consistent results. One of the problem might be in [this function](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/blob/main/inpaint_Lama.py#L179) it seems that sometimes the image does not match the mask and if you pass this image to the LaMa model it make a noisy greyish mess
- [ ] [soft injection](https://github.com/Mikubill/sd-webui-controlnet/blob/7a4805c8ea3256a0eab3512280bd4f84ca0c8182/scripts/hook.py#L620), since I cannot understand how to set different weights inside the sampling steps in comfyUI. At the moment the control seems too high all around, and this cause artifacts in more complex in/outpaint
- [x] LaMa preprocessing node
- [x] Basic workflow

## Thanks

This code is inspired by Mikubill/sd-webui-controlnet
