from tim_reasoning import SessionManager
from os.path import join, dirname

import json


# with open(join(dirname(__file__), '../examples/sample_pinwheels.json'), 'r') as f:
#     data = f.read()


## either this
with open("examples/sample_message_errors.json", "r") as f:
    data = f.read()
jd = json.loads(data)
sm = SessionManager(patience=1)
for j in jd:
    sm.handle_message(message=j['values'])


## or this

# with open("examples/sample_pinwheels.json", "r") as f:
#     data = f.read()

# jd = json.loads(data)
# sm = SessionManager(patience=1)
# sm.handle_message(message=jd)
