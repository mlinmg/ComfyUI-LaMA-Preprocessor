import math

import cv2

# https://github.com/advimman/lama
import logging

import torch

import numpy as np
import os
import sys
from PIL import Image
from scipy import ndimage

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.abspath(os.path.join(my_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

# Append comfy_dir to sys.path & import files
sys.path.append(comfy_dir)
from nodes import LatentUpscaleBy, KSampler, KSamplerAdvanced, VAEDecode, VAEDecodeTiled, VAEEncode, VAEEncodeTiled, \
    ImageScaleBy, CLIPSetLastLayer, CLIPTextEncode, ControlNetLoader, ControlNetApply, ControlNetApplyAdvanced, SetLatentNoiseMask
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL, CLIPTextEncodeSDXLRefiner
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.latent_formats

# Ottieni il percorso assoluto alla radice del tuo progetto
percorso_radice_progetto = os.path.abspath(os.path.dirname(__file__))

# Aggiungi il percorso alla variabile d'ambiente PYTHONPATH
sys.path.append(percorso_radice_progetto)

LOGGER = logging.getLogger(__name__)

from annotator.lama import LamaInpainting


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

model_lama = None
def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

#debug
def show_image(data):
    # Se i dati sono un tensore PyTorch, convertili in un array numpy
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Verifica le dimensioni dei dati
    if len(data.shape) != 3:
        raise ValueError("I dati devono avere 3 dimensioni: (c, h, w)")

    # Ottieni il numero di canali
    c = data.shape[0]

    # Se c è 1, convertiamo l'array in 2D per visualizzarlo come un'immagine in scala di grigi
    if c == 1:
        data = np.squeeze(data)
        mode = 'L'
    elif c == 3:
        # Se c è 3, trasponiamo l'array per avere la forma (h, w, c)
        data = data.transpose(1, 2, 0)
        mode = 'RGB'
    else:
        raise ValueError("Il numero di canali deve essere 1 o 3")

    # Crea un'immagine PIL e visualizzala
    img = Image.fromarray(data.astype('uint8'), mode)
    img.show()


import numpy as np
import torch
from einops import rearrange
import cv2


def safe_numpy(x):
    y = x.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y


def get_pytorch_control(x):
    y = torch.from_numpy(x)
    y = y.float() / 255.0
    y = rearrange(y, 'h w c -> 1 c h w')
    y = y.clone()
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')  # Assumendo che tu abbia definito la variabile 'devices'
    y = y.clone()
    return y


def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]

def nake_nms(x):
    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    y = np.zeros_like(x)
    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
    return y

def remove_pattern(x, kernel):
    objects = cv2.morphologyEx(x, cv2.MORPH_HITMISS, kernel)
    objects = np.where(objects > 127)
    x[objects] = 0
    return x, objects[0].shape[0] > 0


lvmin_kernels_raw = [
    np.array([
        [-1, -1, -1],
        [0, 1, 0],
        [1, 1, 1]
    ], dtype=np.int32),
    np.array([
        [0, -1, -1],
        [1, 1, -1],
        [0, 1, 0]
    ], dtype=np.int32)
]

lvmin_kernels = []
lvmin_kernels += [np.rot90(x, k=0, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=1, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=2, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=3, axes=(0, 1)) for x in lvmin_kernels_raw]

lvmin_prunings_raw = [
    np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [0, 0, -1]
    ], dtype=np.int32),
    np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, 0, 0]
    ], dtype=np.int32)
]

lvmin_prunings = []
lvmin_prunings += [np.rot90(x, k=0, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=1, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=2, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=3, axes=(0, 1)) for x in lvmin_prunings_raw]


def thin_one_time(x, kernels):
    y = x
    is_done = True
    for k in kernels:
        y, has_update = remove_pattern(y, k)
        if has_update:
            is_done = False
    return y, is_done

def lvmin_thin(x, prunings=True):
    y = x
    for i in range(32):
        y, is_done = thin_one_time(y, lvmin_kernels)
        if is_done:
            break
    if prunings:
        y, _ = thin_one_time(y, lvmin_prunings)
    return y

def postprocess(final_inpaint_feed: torch.Tensor, blur_kernel_size: int=7):
    print("doing inpaint only post processing")

    final_inpaint_feed = final_inpaint_feed.detach().cpu().numpy()
    final_inpaint_feed = np.ascontiguousarray(final_inpaint_feed).copy()
    final_inpaint_mask = final_inpaint_feed[0, 3, :, :].astype(np.float32)
    final_inpaint_raw = final_inpaint_feed[0, :3].astype(np.float32)
    sigma = blur_kernel_size
    final_inpaint_mask = cv2.dilate(final_inpaint_mask, np.ones((sigma, sigma), dtype=np.uint8))
    final_inpaint_mask = cv2.blur(final_inpaint_mask, (sigma, sigma))[None]
    _, Hmask, Wmask = final_inpaint_mask.shape
    final_inpaint_raw = torch.from_numpy(np.ascontiguousarray(final_inpaint_raw).copy())
    final_inpaint_mask = torch.from_numpy(np.ascontiguousarray(final_inpaint_mask).copy())
    #mix the images with the mask
    final_inpaint_raw = final_inpaint_raw.to(final_inpaint_mask.dtype).to(final_inpaint_mask.device)
    final_inpaint_mask = final_inpaint_mask.to(final_inpaint_raw.dtype).to(final_inpaint_raw.device)
    final_inpaint_feed = final_inpaint_mask * final_inpaint_raw + (1 - final_inpaint_mask) * final_inpaint_raw
    final_inpaint_feed = final_inpaint_feed.clip(0, 1)
    return final_inpaint_feed


def high_quality_resize(x, size):
    # Written by lvmin
    # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

        # Verifica se l'immagine ha un canale alpha e, in caso affermativo, separalo
    inpaint_mask = None
    if x.ndim == 3 and x.shape[2] == 4:
        inpaint_mask = x[:, :, 3]
        x = x[:, :, 0:3]

    new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
    new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
    unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
    is_one_pixel_edge = False
    is_binary = False
    if unique_color_count == 2:
        is_binary = np.min(x) < 16 and np.max(x) > 240
        if is_binary:
            xc = x
            xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
            xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
            one_pixel_edge_count = np.where(xc < x.squeeze())[0].shape[0]
            all_edge_count = np.where(x > 127)[0].shape[0]
            is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

    if 2 < unique_color_count < 200:
        interpolation = cv2.INTER_NEAREST
    elif new_size_is_smaller:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    # Ridimensiona l'immagine e la maschera, se presente
    y = cv2.resize(x, size, interpolation=interpolation)
    if inpaint_mask is not None:
        inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

    # Se l'immagine è binaria, applica ulteriori trasformazioni
    if is_binary:
        y = y.astype('uint8')
        #y_gray = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        _, y_bin = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        y = cv2.cvtColor(y_bin, cv2.COLOR_GRAY2BGR)

    # Se c'è una maschera, aggiungila all'immagine ridimensionata
    if inpaint_mask is not None:
        inpaint_mask = (inpaint_mask > 127).astype(np.uint8) * 255
        y = np.concatenate([y, inpaint_mask[:, :, None]], axis=2)

    return y
safeint = lambda x: int(np.round(x))


def apply_border_noise(detected_map, outp, mask, h, w):
    # Mantieni solo i primi 3 canali
    detected_map = detected_map[:, :, 0:3].copy()
    detected_map = detected_map.astype(np.float32)
    new_h, new_w, _ = mask.shape
    # Calcola i fattori di ridimensionamento
    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w
    k = min(k0, k1)

    if outp == "outpainting":
        # Trova i pixel sul bordo della maschera
        border_pixels = np.concatenate([detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0)
        # Calcola il colore mediano di questi pixel (ignorando i pixel neri)
        high_quality_border_color = np.median(border_pixels, axis=0).astype(detected_map.dtype)
        # Crea lo sfondo di alta qualità
        high_quality_background = np.tile(high_quality_border_color[None, None], [safeint(h), safeint(w), 1])
    else: #TODO: controlla che i buchi siano creati e riempiti correttamente
            # find the holes in the mask( where is equal to white)
            mask = mask.max(axis=2) > 254  # Adatta questa soglia come necessario
            labeled, num_features = ndimage.label(mask)
            high_quality_background = np.zeros(
                (h, w, 3))  # Crea un'immagine vuota con le stesse dimensioni
            for i in range(1, num_features + 1):
                specific_mask = labeled == i

                # Trova i bordi del buco specifico
                borders = np.concatenate([
                    detected_map[1:, :][specific_mask[1:, :] & ~specific_mask[:-1, :]],
                    detected_map[:-1, :][specific_mask[:-1, :] & ~specific_mask[1:, :]],
                    detected_map[:, 1:][specific_mask[:, 1:] & ~specific_mask[:, :-1]],
                    detected_map[:, :-1][specific_mask[:, :-1] & ~specific_mask[:, 1:]]
                ], axis=0)

                # Calcola il colore mediano per il buco specifico
                high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)

                # Riempie il buco con il colore mediano su un'immagine vuota
                high_quality_background[specific_mask] = high_quality_border_color

            # Assicurati che mask abbia le stesse dimensioni di high_quality_background
    detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
    mask = high_quality_resize(mask,(safeint(w), safeint(h)))
    img_rgba = np.zeros((high_quality_background.shape[0], high_quality_background.shape[1], 4), dtype=np.float32)
    img_rgba[:, :, 0:3] = high_quality_background
    img_rgba[:, :, 3] = 255
    img_rgba_map = np.zeros((detected_map.shape[0], detected_map.shape[1], 4), dtype=np.float32)
    img_rgba_map[:, :, 0:3] = detected_map
    img_rgba_map[:, :, 3] = 0
    detected_map = img_rgba_map
    #add mask.squeeze() to high quality bg to make it 4 channels
    high_quality_background = img_rgba

    new_h, new_w, _ = detected_map.shape
    pad_h = max(0, (h - new_h) // 2)
    pad_w = max(0, (w - new_w) // 2)
    high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = detected_map
    #modify the mask the same way you modify high_quality_background
    detected_map = high_quality_background
    detected_map = safe_numpy(detected_map)
    return get_pytorch_control(detected_map), detected_map


def create_alpha_mask(w, old_w, h, old_h, detected_map):
    # Verifica se detected_map ha un canale alpha; se non lo ha, aggiungilo
    if detected_map.shape[2] < 4:
        alpha_channel = np.zeros((detected_map.shape[0], detected_map.shape[1], 1), dtype=detected_map.dtype)
        detected_map = np.concatenate([detected_map, alpha_channel], axis=2)

    # Creiamo un'immagine di sfondo con il canale alpha impostato a 1 e gli altri canali a 0
    high_quality_background = np.zeros((h, w, 4), dtype=detected_map.dtype)
    high_quality_background[:, :, 3] = 255  # Imposta il canale alpha a 1 (255)

    # Calcola i valori di padding e applica il padding all'immagine originale
    pad_h = max(0, (h - old_h) // 2)
    pad_w = max(0, (w - old_w) // 2)

    # Inserisce l'immagine originale nell'immagine di sfondo (che ha il padding)
    high_quality_background[pad_h:pad_h + old_h, pad_w:pad_w + old_w, :] = detected_map

    return high_quality_background


def decode_latent(latent, vae):
    return VAEDecode().decode(vae, latent)[0]
def unload_lama_inpaint():
    global model_lama
    if model_lama is not None:
        model_lama.unload_model()

def call_vae_using_process(inpaint_model, x, batch_size=None, mask=None):
        try:
            if x.shape[2] > 3:
                x = x[:, :,  0:3]
            x = x * 2.0 - 1.0
            if mask is not None:
                # TODO: throws error if no mask is given
                x = x * (1.0 - mask)

            if inpaint_model is not None:

                vae_output = inpaint_model.encode(inpaint_model,x)
                vae_output = inpaint_model.sd_model.get_first_stage_encoding(vae_output)
                print(f'ControlNet used VAE to encode {vae_output.shape}.')
            else:
                raise AssertionError('No inpaint_model found.')
            latent = vae_output
            if batch_size is not None and latent.shape[0] != batch_size:
                latent = torch.cat([latent.clone() for _ in range(batch_size)], dim=0)
            return latent
        except Exception as e:
            print(e)
            raise ValueError(
                'ControlNet failed to use VAE. Please try to add `--no-half-vae`, `--no-half` and remove `--precision full` in launch cmd.')


class lamaPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pixels": ("IMAGE", ),
                     "type": (["outpainting", "inpainting"], ),
                     "mask": ("MASK",),
                        "vae": ("VAE",),
                     },

                 }


    def preprocess(self,pixels,vae, type,mask=None):

        #mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               #size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
        #mask = rearrange(mask, '1 c h w -> h w c')
        mask = (mask.numpy()*255).astype(np.float32)
        mask = np.expand_dims(mask, -1)
        #mask = high_quality_resize(mask, (((mask.shape[1]) // 8) * 8, ((mask.shape[0]) // 8) * 8))


        pixels = rearrange(pixels, '1 h w c -> h w c')
        pixels = pixels.clone()
        pixels = (pixels.numpy()*255).astype(np.uint8)
        pixels = HWC3(pixels)
        # Crea una maschera booleana dove i pixel non sono neri
        mask_non_black = (mask[:, :, 0] == 0)
        cv2.resize(mask, (((mask.shape[1]) // 8) * 8, ((mask.shape[0]) // 8) * 8), interpolation=3)        # Trova le coordinate dei pixel non neri
        coords = np.column_stack(np.nonzero(mask_non_black))

        # Trova i valori minimo e massimo per x e y
        y_min, y_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        x_min, x_max = np.min(coords[:, 1]), np.max(coords[:, 1])

        # "Croppa" l'immagine per mantenere solo i pixel non neri
        img_non_black = pixels[y_min:y_max + 1, x_min:x_max + 1]
        #Image.fromarray(pixels).save("C:\\Users\marco\Desktop\lama-main\pixels_just_before_lama.png")
        #add a zero channel aplha to pixels
        if type == "outpainting":#TODO: solo un lato alla volta è modificabile, se si vuole modificare anche l'altro lato bisogna fare un altra chiamata
            h = ((mask.shape[0]) // 8) * 8
            w = ((mask.shape[1]) // 8) * 8
            #Image.fromarray(img_non_black[:, :, 0:3].astype('uint8')).save(r"C:\Users\marco\Desktop\moisy\imm_pre.png")
            #pixels = np.concatenate([pixels, mask], axis=2)
            #img_non_black_corr = high_quality_resize(img_non_black, (w, h))

            #pixels = np.concatenate([pixels, mask], axis=2)
            control, img= apply_border_noise(img_non_black, type, mask, h, w)
        elif type == "inpainting":
            if mask is None:
                raise ValueError("If you want to use inpainting you must provide a mask")
            h = (pixels.shape[2])
            w = (pixels.shape[1])
            _, img = apply_border_noise(pixels, type, mask, h, w)
        else:
            raise ValueError("Type must be outpainting or inpainting")

        #imag_noise = noise_hack(pixels, vae)
        H, W, C = img.shape
        #raw_color = img[:, :, 0:3]#test
        raw_mask = img[:, :, 3:4]#test

        res = 256  # Always use 256 since lama is trained on 256
        #TODO: questa restituisce i bordi troppo chiari, controllare
        image_res, remove_pad = resize_image_with_pad(img, res, skip_hwc3=True)

        global model_lama
        if model_lama is None:
            model_lama = LamaInpainting()

        # applied auto inversion
        try:
            prd_color = model_lama(image_res)
            #model_lama.unload_model()
        except Exception as e:
            print(e)
            raise e
        prd_color = remove_pad(prd_color)
        prd_color = cv2.resize(prd_color, (W, H))
        mask_alpha = raw_mask > 0
        # Dilata la maschera
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask_alpha.astype(np.uint8), kernel)
        # Crea una maschera di sfumatura
        mask_blur = cv2.GaussianBlur(mask_dilated.astype(np.float32), (21, 21), 0)
        # Ripete la maschera di sfumatura per tutti e 3 i canali
        mask_blur_rgb = np.repeat(mask_blur[..., np.newaxis], 3, axis=2)
        input_img_resized = high_quality_resize(pixels, (W, H))
        # Utilizza la maschera di sfumatura in np.where per creare un effetto di sfumatura
        final_img = np.where(mask_blur_rgb > 0.5, prd_color[..., :3], input_img_resized[:, :, 0:3])
        # Aggiungi il canale alpha di high_quality_background al risultato finale

        final_img = get_pytorch_control(final_img)
        final_img = rearrange(final_img, ('1 c h w -> 1 h w c'))
        raw_mask = torch.from_numpy(mask_blur) #raw_mask = rearrange(torch.from_numpy(np.ascontiguousarray(mask_blur_rgb).copy()) /255.0,('h w c -> 1 h w c'))
        image = final_img.clone()
        #image[raw_mask > 0.5] = -1.0  # set as masked pixel
        encoder = VAEEncode()
        set_latent_mask = SetLatentNoiseMask()
        encoded_image = encoder.encode(vae=vae, pixels=image)[0] #TODO: sembra corretto
        #inserisci rumore nel latente
        #create a mask with the same size of mask but all ones
        mas = rearrange(raw_mask.unsqueeze(-1),('h w c -> 1 c h w'))
        latent_mask = encoded_image#set_latent_mask.set_mask(encoded_image, raw_mask)[0] #TODO: la maschera non sembra definita correttamente, controllare
        return (image, latent_mask, mas)

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")
    RETURN_NAMES = ("LaMa Preprocessed Image", "LaMa Preprocessed Latent", "LaMa Preprocessed Mask")
    FUNCTION = "preprocess"
    CATEGORY = "image/preprocessors"
NODE_CLASS_MAPPINGS = {
    "LaMaPreprocessor": lamaPreprocessor,}