import json, sys

# Deliberate negative control. It repeats the first advertised action.
first_action = None
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "episode_start":
        actions = msg.get("legal_actions", [])
        first_action = actions[0] if actions else "__invalid__"
    print(json.dumps({"action": first_action}), flush=True)
