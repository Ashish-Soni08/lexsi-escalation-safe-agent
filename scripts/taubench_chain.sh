#!/usr/bin/env bash
# tau-bench retail (115 test tasks, one trial), reduced form: the agent is one of our models behind vLLM, driven
# with tau-bench's text "react" strategy; the simulated customer is an open model, not GPT-4o. Not comparable with
# the public leaderboard; it compares our models with each other on a public task set.
#   scripts/taubench_chain.sh "<name=repo> <name=repo> ..."
set -uo pipefail
R=results/taubench; mkdir -p $R
USER_MODEL=${USER_MODEL:-Qwen/Qwen2.5-14B-Instruct}
python3 -m venv /tb && /tb/bin/pip install -q "git+https://github.com/sierra-research/tau-bench" || exit 1
SITE=$(/tb/bin/python -c "import tau_bench, os; print(os.path.dirname(tau_bench.__file__))")
# litellm has no price for a local model and returns None as the cost; tau-bench then fails on the sum
sed -i 's/res._hidden_params\["response_cost"\]$/(res._hidden_params.get("response_cost") or 0.0)/' $SITE/agents/chat_react_agent.py $SITE/envs/user.py
sed -i 's/self.total_cost = res._hidden_params\["response_cost"\]/self.total_cost = res._hidden_params.get("response_cost") or 0.0/' $SITE/envs/user.py
grep -n "response_cost" $SITE/agents/chat_react_agent.py $SITE/envs/user.py
up() {  # <model> <port> <served name> <gpu share> <log>
  vllm serve "$1" --port "$2" --served-model-name "$3" --gpu-memory-utilization "$4" --max-model-len 8192 > "$5" 2>&1 &
  echo $! > /tmp/vllm_$2.pid
  for i in $(seq 1 240); do python3 -c "import urllib.request as u; u.urlopen('http://localhost:$2/health', timeout=3)" 2>/dev/null && return 0; sleep 5; done
  echo "TAU FAILED server $1"; tail -30 "$5"; return 1
}
up "$USER_MODEL" 8001 usersim 0.50 $R/vllm_user.log || exit 1
export OPENAI_API_BASE=http://localhost:8001/v1 OPENAI_API_KEY=none HOSTED_VLLM_API_BASE=http://localhost:8000/v1 HOSTED_VLLM_API_KEY=none
for PAIR in $1; do
  NAME=${PAIR%%=*}; REPO=${PAIR#*=}
  up "$REPO" 8000 agent 0.35 $R/vllm_$NAME.log || continue
  /tb/bin/python - <<P || echo "TAU FAILED $NAME"
import sys
sys.argv = ["run", "--env", "retail", "--task-split", "test", "--agent-strategy", "react", "--model", "agent",
            "--model-provider", "hosted_vllm", "--user-model", "usersim", "--user-model-provider", "openai",
            "--user-strategy", "llm", "--max-concurrency", "16", "--log-dir", "$R/$NAME"] + "${TAU_ARGS:-}".split()
import runpy, importlib.util, os, urllib.request
src = urllib.request.urlopen("https://raw.githubusercontent.com/sierra-research/tau-bench/main/run.py").read().decode()
exec(compile(src, "run.py", "exec"), {"__name__": "__main__"})
P
  kill $(cat /tmp/vllm_8000.pid); sleep 15
done
kill $(cat /tmp/vllm_8001.pid)
python3 scripts/taubench_report.py $R
