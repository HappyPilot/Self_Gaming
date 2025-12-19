import sys, json
sys.path.insert(0, '/home/dima/self-gaming/agents')
from mem_rpc import MemRPC
rpc = MemRPC()
resp = rpc.query({"mode": "calibration_events", "scope": "critical_dialog:death", "limit": 5}, timeout=2.0)
print(json.dumps(resp, indent=2))
rpc.close()
