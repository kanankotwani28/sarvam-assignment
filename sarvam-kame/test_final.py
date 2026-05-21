import re

def _clean(text):
    think_open = '<think'
    think_close = '/think>'
    
    # Case 1: Complete <think>... block - remove it
    pattern1 = think_open + '.*?' + think_close
    text = re.sub(pattern1, '', text, flags=re.DOTALL).strip()
    
    # Case 2: Incomplete thinking (no closing tag) - find where actual response starts
    # Look for a period/exclamation followed by uppercase (start of new sentence)
    match = re.search(r'<think>.*?([.!?]\s+[A-Z])', text, re.DOTALL)
    if match:
        text = text[match.end():].strip()
    else:
        # If no sentence boundary found, just remove the <think> line itself
        text = re.sub(r'<think>.*$', '', text, flags=re.MULTILINE).strip()
    
    # Case 3: Any remaining reasoning tags
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()
    
    if not text.strip():
        text = 'I am ready to help.'
    return text

# Test 1: truncated response (no closing tag)
t1 = '<think>\nOkay, the user said hello. I should respond.'
print('Test 1:', repr(_clean(t1)))

# Test 2: complete response with closing tag
t2 = '<think>\nThinking...\n\n\nHello there!'
print('Test 2:', repr(_clean(t2)))

# Test 3: normal response (no thinking)
t3 = 'Hello! How can I help?'
print('Test 3:', repr(_clean(t3)))

# Test 4: Real API response format
t4 = '''<think>
Okay, the user says "Hello, my name is Kanan." I need to respond appropriately.

First, I should greet them back. Maybe say "Hello Kanan, nice to meet you!" That's friendly and short.

Since the user mentioned'''
print('Test 4:', repr(_clean(t4)))