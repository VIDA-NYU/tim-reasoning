from tim_reasoning import SessionManager
from os.path import join, dirname

import json


with open(join(dirname(__file__), '../examples/sample_message.json'), 'r') as f:
    data = f.read()

jd = json.loads(data)
sm = SessionManager(patience=1)
# for j in jd:
# sm.handle_message(message=j['values'])
sm.handle_message(message=jd)
