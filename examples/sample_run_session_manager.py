from tim_reasoning import SessionManager

import json

with open('examples/sample_pinwheels.json', 'r') as f:
    data = f.read()

jd = json.loads(data)
sm = SessionManager(patience=1)
# for j in jd:
# sm.handle_message(message=j['values'])
sm.handle_message(message=jd)
