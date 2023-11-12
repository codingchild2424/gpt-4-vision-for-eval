import gradio as gr
import cv2
import base64
import openai

def process_video(video_file, api_key, instruction):
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

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                instruction,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
            ],
        },
    ]

    try:
        result = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=PROMPT_MESSAGES,
            api_key=openai.api_key,
            max_tokens=500,
        )
        return result.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
    
# Define the Gradio app
def main():
    with gr.Blocks() as app:
        gr.Markdown("## GPT-4 Vision for Evaluation")
        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(label="Enter your OpenAI API Key", lines=1)
                instruction_input = gr.Textbox(label="Enter Narration Instruction", placeholder="Enter your custom instruction here...", lines=5)
                video_upload = gr.File(label="Upload your video", type="file")
                submit_button = gr.Button("Generate Script")
            with gr.Column(scale=1):
                output_box = gr.Textbox(label="Generated Script", lines=7, interactive=False)

        submit_button.click(fn=process_video, inputs=[video_upload, api_key_input, instruction_input], outputs=output_box)

    app.launch()

if __name__ == "__main__":
    main()