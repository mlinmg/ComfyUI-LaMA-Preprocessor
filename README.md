# LaMa Preprocessor (WIP)

Currenly only supports NVIDIA

This preprocessor finally enable users to generate coherent inpaint and outpaint **prompt-free**

For inpainting tasks, it's recommended to use the 'outpaint' function. Although the 'inpaint' function is still in the development phase, the results from the 'outpaint' function remain quite satisfactory.

The best results are given on landscapes, good results can still be achieved in drawings by lowering the controlnet end percentage to 0.7-0.8

## Installation

1. Manager installation: be sure to have [ComfyUi Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, then just search for lama preprocessor
  
2. Manual Installation: clone this repo inside the custom_nodes folder
  
  ## A LaMa prerocessor for ComfyUi
  
  This is a simple workflow example.To use this, download workflows/workflow_lama.json and then drop it in a ComfyUI tab
  

---

This are some non cherry picked results, all obtained starting from this image

---

You can find the processor in image/preprocessors

## Contributing

Everyone is invited to contribute, to do so you can just make a pull request

If you would like to help to the development of this repo there are some missing features that still need to be implemented:

- [x] An unusual behavior is observed when providing an empty prompt to the drawing/cartoon outpainting system. Interestingly, the results appear to be significantly better when the two conditionings are prompted with "positive" and "negative" respectively. This is probably beacause when prompted with a blank prompt, the controlnet adds too much weights.
- [ ] This workflow exibits some image darkening/color shifting, this should be further investigated in order to apply a color fix method
- [ ] More consistent results. ~~One of the problem might be in [this function](https://github.com/mlinmg/ComfyUI-LaMA-Preprocessor/blob/main/inpaint_Lama.py#L179) it seems that sometimes the image does not match the mask and if you pass this image to the LaMa model it make a noisy greyish mess~~ this has been ruled out since the auto1111 preprocess gives approximately the same image as in comfyui. bit the consistency problem remain and the results are really different when compared to the automatic1111 repo. for anyone interested in contributing I have already inplemented a soft injection mechanism, you should start from here to see where it goes south. One of the problems I have seen is the difference in the clip embeddings
- [x] [soft injection](https://github.com/Mikubill/sd-webui-controlnet/blob/7a4805c8ea3256a0eab3512280bd4f84ca0c8182/scripts/hook.py#L620),~~since I cannot understand how to set different weights inside the sampling steps in comfyUI. At the moment the control seems too high all around, and this cause artifacts in more complex in/outpaint~~ while I've been able to succesfully implement this, it seems to lower the weights of controlnet too much, using a soft injection will lead to broken results
- [x] LaMa preprocessing node
- [x] Basic workflow

## Thanks

This code is inspired by Mikubill/sd-webui-controlnet
