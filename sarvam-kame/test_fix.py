import re
text = "<think>\nOkay, the user said hello.\n</think>\n\nHello Kanan! Nice to meet you."
clean = re.sub(r'<think>.*?
</think>

', '', text, flags=re.DOTALL).strip()
print('CLEAN:', clean)
