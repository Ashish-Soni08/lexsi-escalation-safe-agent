#!/usr/bin/env bash
# Run a command on HF Jobs against the current code + data snapshot.
#   scripts/job.sh <tag> <flavor> <timeout> <image> "<command run inside the repo dir>"
# Uploads code/ to the private workdir dataset, runs the command in the job,
# then uploads ./results and ./outputs from the job to runs/<tag>/ in the workdir.
# ESA_PRE="<command>;" runs first inside the job (e.g. install the hf CLI in an image that lacks it).
# ESA_NAMESPACE=<org> runs and bills the job in that organisation (the token needs Jobs write there).
set -euo pipefail
TAG=$1; FLAVOR=$2; TIMEOUT=$3; IMAGE=$4; CMD=$5
# ESA_WORKDIR=<user>/<dataset> picks the scratch dataset; by default <your HF user>/esa-workdir (created private if missing).
WD=${ESA_WORKDIR:-$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")/esa-workdir}
hf repo create "$WD" --repo-type dataset --private --exist-ok >/dev/null 2>&1 || true
cd "$(dirname "$0")/.."
hf upload "$WD" . code --repo-type dataset --exclude ".git/*" --exclude "results/*" --exclude "outputs/*" --exclude "*__pycache__*" --commit-message "code for $TAG" >/dev/null
INNER="set -x; ${ESA_PRE:-} \
hf download $WD --repo-type dataset --include 'code/*' --local-dir /w >/dev/null && cd /w/code && mkdir -p results outputs && \
( $CMD ) 2>&1 | tee results/job_${TAG}.log; rc=\${PIPESTATUS[0]}; \
hf upload $WD results runs/$TAG/results --repo-type dataset >/dev/null; \
[ -n \"\$(ls -A outputs)\" ] && hf upload $WD outputs runs/$TAG/outputs --repo-type dataset >/dev/null; exit \$rc"
hf jobs run --detach ${ESA_NAMESPACE:+--namespace "$ESA_NAMESPACE"} --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN --name "esa-$TAG" "$IMAGE" bash -c "$INNER"
