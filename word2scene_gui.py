import configparser
import os
import re
import webuiapi
from PIL import PngImagePlugin
import gradio as gr

label_map = {
    'en': {
        'title': '## Word2Scene: Efficient remote sensing image scene generation with only one word via hybrid intelligence and low-rank representation',
        'advanced_options': 'Advanced options',
        'prompt': 'Knowledge',
        'prompt_placeholder': 'Enter the description of the remote sensing scene you want to generate, you can enter the complete sentence or just the RS scene concept, such as: resort',
        'generate_button': 'Generate',
        'output_dir': 'Output path',
        'output_dir_placeholder': 'Path to save the generated RS scene images (absolute path), if not entered, the generated RS scene images will not be saved',
        'image_size': 'Image size',
        'num_samples': 'Num samples',
        'steps': 'Steps',
        'cfg_scale': 'Knowledge strength',
        'denoise_strength': 'Denoise strength',
        'seed': 'Seed',
        'single_image_gallery': 'Generated results',
    }
}


def generate_scene_image(prompt, image_size, denoise_strength, steps, num_samples, cfg_scale, seed, output_dir):
    '''
    Generate RS scene images based on the provided parameters.

    Parameters:
    prompt (str): The prompt to generate the image.
    image_size (int): The size of the generated image.
    denoise_strength (float): The strength of the denoising applied to the generated image.
    steps (int): The number of steps to use for the generation process.
    num_samples (int): The number of samples to generate.
    cfg_scale (float): The scale of the configuration used for the generation process.
    seed (int): The seed used for the random number generator.
    output_dir (str): The directory where the generated images will be saved.

    Returns:
    list: A list of the generated images.
    '''
    full_prompt = prompt + ",<lora:word2scene:1>"
    result = api.txt2img(prompt=full_prompt,
                         negative_prompt=negative_prompt,
                         seed=seed,
                         styles=[],
                         cfg_scale=cfg_scale,
                         width=image_size,
                         height=image_size,
                         batch_size=num_samples,
                         n_iter=1,
                         steps=steps,
                         denoising_strength=denoise_strength
                         )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # 将result.images中的所有图像保存到本地
        for i in range(len(result.images)):
            # 获取生成图片的信息,并保存到图片中
            img_info = PngImagePlugin.PngInfo()
            img_info.add_text("parameters", result.images[i].info['parameters'])
            img_seed = re.search(r'Seed: (\d+)', result.images[i].info['parameters']).group(1)
            result.images[i].save(os.path.join(output_dir, prompt[:40] + '-' + img_seed + '-' + str(i) + ".png"),
                                  pnginfo=img_info)
    return result.images


if __name__ == '__main__':
    api = webuiapi.WebUIApi(port=7860, sampler='DDIM', steps=30)
    cf = configparser.ConfigParser()
    cf.read("config/configs.json")

    server_name = cf.get("server", "name")
    server_port = cf.getint("server", "port")
    negative_prompt = cf.get("generate", "negative_prompt")
    user_locale = cf.get("generate", "user_locale")
    # create API client with default sampler, steps.
    api = webuiapi.WebUIApi(host=cf.get("generate", "host"), port=cf.getint("generate", "port"),
                            sampler=cf.get("generate", "sampler"), steps=cf.getint("generate", "steps"))

    with gr.Blocks(title="Word2Sence").queue() as block:
        with gr.Row():
            gr.Markdown(label_map[user_locale]['title'])
        with gr.TabItem('Word2Scene', id='generate_single_image',
                        elem_id="word2scene_single_tab") as tab_generate_single_image:
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label=label_map[user_locale]['prompt'], visible=True, value="",
                                        placeholder=label_map[user_locale]['prompt_placeholder'])
                    generate_button = gr.Button(value=label_map[user_locale]['generate_button'])
                    with gr.Accordion(label=label_map[user_locale]['advanced_options'], open=False):
                        output_dir = gr.Textbox(label=label_map[user_locale]['output_dir'], visible=True, value="",
                                                placeholder=label_map[user_locale]['output_dir_placeholder'],
                                                elem_id="single_output_dir")
                        image_size = gr.Slider(label=label_map[user_locale]['image_size'], minimum=128, maximum=1024,
                                               value=512, step=1)
                        num_samples = gr.Slider(
                            label=label_map[user_locale]['num_samples'], minimum=1, maximum=4, value=1, step=1)
                        steps = gr.Slider(label=label_map[user_locale]['steps'], minimum=1, maximum=50, value=30,
                                          step=1)
                        cfg_scale = gr.Slider(
                            label=label_map[user_locale]['cfg_scale'], visible=True, minimum=0.1, maximum=30.0,
                            value=7.0, step=0.1
                        )
                        denoise_strength = gr.Slider(label=label_map[user_locale]['denoise_strength'], minimum=0,
                                                     maximum=1, value=0.7, step=0.1)
                        seed = gr.Slider(
                            label=label_map[user_locale]['seed'],
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1,
                        )
                with gr.Column():
                    single_image_gallery = gr.Gallery(label=label_map[user_locale]['single_image_gallery'],
                                                      show_label=True).style(columns=2, height="auto")

            generate_button.click(fn=generate_scene_image,
                                  inputs=[prompt, image_size, denoise_strength, steps, num_samples, cfg_scale,
                                          seed, output_dir], outputs=[single_image_gallery])

    block.launch(inbrowser=True, server_name=server_name, server_port=server_port)
