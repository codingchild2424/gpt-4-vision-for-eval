import gradio as gr
import cv2
import base64
import openai
import tempfile

def process_video(video_file, api_key):
    # Set the OpenAI API key
    openai.api_key = api_key

    # Read and process the video file
    video = cv2.VideoCapture(video_file.name)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()

    # Instruction for narration generation
    INSTRUCTION = " ".join([
        "These are frames of a video.",
        "Create a short voiceover script in the style of a super excited Brazilian sports narrator who is narrating his favorite match.",
        "He is a big fan of Messi, the player who scores in this clip.",
        "Use caps and exclamation marks where needed to communicate excitement.",
        "Only include the narration, your output must be in English.",
        "When the ball goes into the net, you must scream GOL either once or multiple times."
    ])

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                INSTRUCTION,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
            ],
        },
    ]

    try:
        result = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=PROMPT_MESSAGES,
            api_key=openai.api_key,
            headers={"Openai-Version": "2020-11-07"},
            max_tokens=500,
        )
        return result.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
    
# Define the Gradio app
def main():
    with gr.Blocks() as app:
        gr.Markdown("## Video Narration Generator")
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(label="Enter your OpenAI API Key")
                video_upload = gr.File(label="Upload your video")
                submit_button = gr.Button("Generate Script", elem_id="submit_button")
            with gr.Column():
                output_box = gr.Textbox(label="Generated Script", lines=10, interactive=False)

        submit_button.click(fn=process_video, inputs=[video_upload, api_key_input], outputs=output_box)

    app.launch()

if __name__ == "__main__":
    main()

# # Define the Gradio app
# def main():
#     with gr.Blocks() as app:
#         gr.Markdown("## Video Narration Generator")
#         with gr.Row():
#             video_upload = gr.File(label="Upload your video")
#             api_key_input = gr.Textbox(label="Enter your OpenAI API Key")
#             submit_button = gr.Button("Generate Script")
#         output_box = gr.Textbox(label="Generated Script", lines=10, interactive=False)

#         submit_button.click(fn=process_video, inputs=[video_upload, api_key_input], outputs=output_box)

#     app.launch()

# if __name__ == "__main__":
#     main()
