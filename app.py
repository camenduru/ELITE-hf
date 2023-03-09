#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr

from model import Model

repo_dir = pathlib.Path(__file__).parent


def create_demo():
    DESCRIPTION = '# [ELITE](https://github.com/csyxwei/ELITE)'

    model = Model()

    with gr.Blocks(css=repo_dir / 'style.css') as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    image = gr.Image(label='Input', tool='sketch', type='pil')
                    gr.Markdown('Draw a mask on your object.')
                prompt = gr.Text(
                    label='Prompt',
                    placeholder='e.g. "A photo of S", "S wearing sunglasses"',
                    info='Use "S" for your concept.')
                lambda_ = gr.Slider(
                    label='Lambda',
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.6,
                    info=
                    'The larger the lambda, the more consistency between the generated image and the input image, but less editability.'
                )
                run_button = gr.Button('Run')
                with gr.Accordion(label='Advanced options', open=False):
                    seed = gr.Slider(
                        label='Seed',
                        minimum=-1,
                        maximum=1000000,
                        step=1,
                        value=-1,
                        info=
                        'If set to -1, a different seed will be used each time.'
                    )
                    guidance_scale = gr.Slider(label='Guidance scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=5.0)
                    num_steps = gr.Slider(
                        label='Steps',
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=100,
                        info=
                        'In the paper, the number of steps is set to 100, but in this demo the default value is 20 to reduce inference time.'
                    )
            with gr.Column():
                result = gr.Image(label='Result')

        paths = sorted([
            path.as_posix()
            for path in (repo_dir / 'ELITE/test_datasets').glob('*')
            if 'bg' not in path.stem
        ])
        gr.Examples(examples=paths, inputs=image, examples_per_page=20)

        inputs = [
            image,
            prompt,
            seed,
            guidance_scale,
            lambda_,
            num_steps,
        ]
        prompt.submit(fn=model.run, inputs=inputs, outputs=result)
        run_button.click(fn=model.run, inputs=inputs, outputs=result)
    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue(api_open=False).launch()
