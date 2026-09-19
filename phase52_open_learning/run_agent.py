import json
import sys
from .mirror7_open_learner import OpenEndedLearner

learner = OpenEndedLearner()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
        response = learner.handle(msg)
        print(json.dumps(response, sort_keys=True), flush=True)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), flush=True)
