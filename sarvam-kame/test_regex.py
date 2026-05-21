import re

def _clean(text):
    think_open = '<think'
    think_close = '/think>'
    pattern = think_open + '.*?(?:' + think_close + '|$)'
    text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    if not text.strip():
        text = 'I am ready to help.'
    return text

# Test 1: truncated response (no closing tag)
t1 = '<think>\nOkay, the user said hello.\nI should respond'
print('Test 1:', repr(_clean(t1)))

# Test 2: complete response with closing tag
t2 = '<think>\nThinking...\n</think>\n\nHello there!'
print('Test 2:', repr(_clean(t2)))

# Test 3: normal response (no thinking)
t3 = 'Hello! How can I help?'
print('Test 3:', repr(_clean(t3)))
