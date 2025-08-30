import os
import gc
import torch
import gradio as gr
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from torch.cuda import OutOfMemoryError

MODEL_DIR = "/home/nyarch/IMAGEGEN/model"
VAE_DIR = os.path.join(MODEL_DIR, "vaes")

loaded_pipes = {}
current_image = None
current_params = {}

def hard_cleanup():
    global loaded_pipes, current_image
    for key in list(loaded_pipes.keys()):
        try:
            if loaded_pipes[key] is not None:
                loaded_pipes[key].to('cpu')
                del loaded_pipes[key]
        except:
            pass
    loaded_pipes = {}
    current_image = None
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()
    torch.cuda.synchronize()

def load_vae(pipe, vae_file):
    """Подгрузка кастомного VAE в pipe"""
    if vae_file is None or vae_file == "Дефолтный VAE":
        return pipe  # оставить дефолтный VAE
    vae_path = os.path.join(VAE_DIR, vae_file)
    if not os.path.isfile(vae_path):
        raise ValueError(f"Файл VAE не найден: {vae_path}")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, local_files_only=True)
    pipe.vae = vae.to(pipe.device)
    return pipe

def load_pipe(model_file, vae_file, use_xformers, use_cpu_offload, use_vae_slicing, use_attention_slicing, img2img=False):
    try:
        hard_cleanup()
        pipe_key = f"{model_file}_{vae_file}_{'img2img' if img2img else 'txt2img'}"
        if pipe_key in loaded_pipes:
            pipe = loaded_pipes[pipe_key]
            pipe.to('cuda')
        else:
            model_path = os.path.join(MODEL_DIR, model_file)
            if img2img:
                pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=True,
                    safety_checker=None
                )
            else:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    local_files_only=True,
                    safety_checker=None
                )
            pipe = load_vae(pipe, vae_file)  # подключаем кастомный VAE или дефолтный

            if use_xformers:
                pipe.enable_xformers_memory_efficient_attention()
            if use_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to('cuda')
            if use_vae_slicing:
                pipe.enable_vae_slicing()
            if use_attention_slicing:
                pipe.enable_attention_slicing()

            loaded_pipes[pipe_key] = pipe

        return pipe
    except Exception as e:
        hard_cleanup()
        raise gr.Error(f"Ошибка загрузки модели: {str(e)}")

def generate_image(*args):
    global current_image, current_params
    torch.cuda.synchronize()

    current_params = {
        'model_file': args[0],
        'vae_file': args[1],
        'prompt': args[2],
        'negative_prompt': args[3],
        'height': int(args[4]),
        'width': int(args[5]),
        'steps': int(args[6]),
        'guidance_scale': float(args[7]),
        'use_xformers': args[8],
        'use_cpu_offload': args[9],
        'use_vae_slicing': args[10],
        'use_attention_slicing': args[11],
        'seed': int(args[12]),
        'denoising_strength': 0.7
    }

    pipe = load_pipe(
        current_params['model_file'],
        current_params['vae_file'],
        current_params['use_xformers'],
        current_params['use_cpu_offload'],
        current_params['use_vae_slicing'],
        current_params['use_attention_slicing'],
        img2img=False
    )
    generator = torch.Generator(device="cuda").manual_seed(current_params['seed'])

    try:
        output = pipe(
            prompt=current_params['prompt'],
            negative_prompt=current_params['negative_prompt'],
            height=current_params['height'],
            width=current_params['width'],
            num_inference_steps=current_params['steps'],
            guidance_scale=current_params['guidance_scale'],
            generator=generator,
            output_type="pil"
        )
        image = output.images[0]
        current_image = image.copy()
        pipe.to('cpu')
        torch.cuda.empty_cache()
        return image, gr.update(visible=True)
    except OutOfMemoryError:
        hard_cleanup()
        raise gr.Error("Не хватает VRAM! Попробуйте:\n1. Уменьшить разрешение\n2. Включить CPU offload\n3. Включить VAE slicing")

def edit_image(edit_prompt, edit_neg_prompt, denoising_strength, edit_steps, edit_guidance, edit_seed):
    global current_image, current_params

    if current_image is None:
        raise gr.Error("Сначала сгенерируйте изображение!")

    try:
        if not isinstance(current_image, Image.Image):
            raise gr.Error(f"Неожиданный тип изображения: {type(current_image)}")
        img_copy = current_image.copy()
        if img_copy.mode != 'RGB':
            img_copy = img_copy.convert('RGB')
    except Exception as e:
        raise gr.Error(f"Ошибка подготовки изображения: {str(e)}")

    pipe = load_pipe(
        current_params['model_file'],
        current_params['vae_file'],
        current_params['use_xformers'],
        current_params['use_cpu_offload'],
        current_params['use_vae_slicing'],
        current_params['use_attention_slicing'],
        img2img=True
    )
    generator = torch.Generator(device="cuda").manual_seed(int(edit_seed))

    try:
        result = pipe(
            prompt=edit_prompt,
            negative_prompt=edit_neg_prompt,
            image=img_copy,
            strength=float(denoising_strength),
            num_inference_steps=int(edit_steps),
            guidance_scale=float(edit_guidance),
            generator=generator,
            output_type="pil"
        )
        edited_image = result.images[0]
        current_image = edited_image.copy()
        pipe.to('cpu')
        torch.cuda.empty_cache()
        return edited_image
    except Exception as e:
        hard_cleanup()
        raise gr.Error(f"Ошибка в процессе редактирования: {str(e)}")

# Получаем список моделей
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith((".safetensors", ".pt", ".bin"))]

# Получаем список VAE, добавляем дефолтный вариант
vae_files = ["Дефолтный VAE"]  # если выбрать, будет использоваться дефолтный VAE из модели
if os.path.isdir(VAE_DIR):
    vae_files += [f for f in os.listdir(VAE_DIR) if f.endswith((".safetensors", ".pt", ".bin"))]

with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion XL Generator (img2img режим)")

    with gr.Row():
        model_select = gr.Dropdown(model_files, label="Выбор модели", value=model_files[0] if model_files else None)
        vae_select = gr.Dropdown(vae_files, label="Выбор VAE", value=vae_files[0])
        seed_input = gr.Number(label="Seed", value=42, precision=0)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", lines=3, value="extremly sexual, same style, keep style, detailed eyes, thick fur, fluffy fur, beautiful, ultra high detailed 8k, score_9, teen furry dog girl, beautiful pretty face, slim young body, very big hips and chest, very thin bikini, ultra provocative pose, thin straps, snug fit, chest forward, anime style, very sexy, slutty attitude, playful, teasing, flirty")
            neg_prompt_input = gr.Textbox(label="Negative Prompt", lines=3, value="different style, makeup, bold, human, red eyes, red lips, mutated bad anatomy, old, broken bikini, mutated bikini, anime, 2D, extra tails, strings, short clothes, drawn, blur, lowres, bad anatomy, modesty, censored, flat shading, extra limbs, old, mature, fat, clothes, boring, plain")

            with gr.Row():
                height_slider = gr.Slider(256, 2048, step=64, label="Высота", value=1024)
                width_slider = gr.Slider(256, 2048, step=64, label="Ширина", value=1024)

            with gr.Row():
                steps_slider = gr.Slider(1, 150, step=1, label="Steps", value=30)
                guidance_slider = gr.Slider(1, 20, step=0.1, label="Guidance scale", value=7.0)

            with gr.Row():
                xformers_checkbox = gr.Checkbox(label="xformers", value=True)
                cpu_offload_checkbox = gr.Checkbox(label="CPU offload", value=True)
                vae_slicing_checkbox = gr.Checkbox(label="VAE slicing", value=True)
                attention_slicing_checkbox = gr.Checkbox(label="Attention slicing", value=True)

            generate_button = gr.Button("Сгенерировать")

        output_image = gr.Image(label="Результат", type="pil")

    with gr.Column(visible=False) as edit_panel:
        with gr.Row():
            with gr.Column():
                edit_prompt = gr.Textbox(label="Промпт для редактирования", lines=3)
                edit_neg_prompt = gr.Textbox(label="Негативный промпт", lines=3)

                with gr.Row():
                    denoise_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Сила изменений")
                    edit_steps = gr.Slider(1, 150, step=1, label="Steps", value=30)
                    edit_guidance = gr.Slider(1, 20, step=0.1, label="Guidance scale", value=7.0)
                    edit_seed = gr.Number(label="Новый Seed", value=42, precision=0)

                edit_button = gr.Button("Применить правки")

            edit_output = gr.Image(label="Редактированный результат", type="pil")

    generate_button.click(
        fn=generate_image,
        inputs=[
            model_select, vae_select,
            prompt_input, neg_prompt_input,
            height_slider, width_slider,
            steps_slider, guidance_slider,
            xformers_checkbox, cpu_offload_checkbox,
            vae_slicing_checkbox, attention_slicing_checkbox,
            seed_input
        ],
        outputs=[output_image, edit_panel]
    )

    edit_button.click(
        fn=edit_image,
        inputs=[edit_prompt, edit_neg_prompt, denoise_slider, edit_steps, edit_guidance, edit_seed],
        outputs=edit_output
    )


demo.launch(server_name="0.0.0.0", server_port=25565)
