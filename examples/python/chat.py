"""Chat with any GGUF model:   python chat.py MODEL.gguf "your question" """
import sys

import dooms

if len(sys.argv) < 3:
    sys.exit(__doc__)
llm = dooms.LLM(sys.argv[1])
for piece in llm.stream_chat([{"role": "user", "content": sys.argv[2]}], max_new_tokens=128):
    print(piece, end="", flush=True)
print()
