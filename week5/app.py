from ragllmagent.agent import chat
import gradio as gr

if __name__ == "__main__":
    gr.ChatInterface(fn=chat).launch(inbrowser=True)
