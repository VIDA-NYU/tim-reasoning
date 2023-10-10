from tim_reasoning import SessionManager

import json

with open('examples/sample_message.json', 'r') as f:
    data = f.read()

jd = json.loads(data)
sm = SessionManager()
for j in jd:
    sm.handle_message(message=j['values'])
