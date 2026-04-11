"""
chat.py — Text chat with IRA. Maintains conversation history.

Run:
    uv run python chat.py
"""

from inference import generate_stream

print("IRA chat — type 'quit' to exit\n")

history = []

while True:
    try:
        msg = input("you: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")
        break

    if not msg or msg.lower() in {"quit", "exit"}:
        break

    print("IRA: ", end="", flush=True)
    reply_tokens = []
    for tok in generate_stream(msg, history=history):
        print(tok, end="", flush=True)
        reply_tokens.append(tok)
    print("\n")

    reply = "".join(reply_tokens)
    history.append({"role": "user",      "content": msg})
    history.append({"role": "assistant", "content": reply})
