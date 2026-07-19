import glob
import tiktoken

MODEL='gpt-4.1-mini'

def print_kb_details():
    kb=''
    filenames=glob.glob('week5/knowledge-base/**/*.md')
    for filename in filenames:
        with open(file=filename,mode='r',encoding='utf-8') as file_handle:
            kb +=file_handle.read() + '\n\n'
    tokens=tiktoken.encoding_for_model(MODEL).encode(kb)
    print(f'kb size: {len(kb)}')
    print(f'token count: {len(tokens)}')


