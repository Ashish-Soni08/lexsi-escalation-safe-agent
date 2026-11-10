"""Submit shell jobs to Ashish's deployed Modal app over HTTPS and poll them.

The Modal CLI cannot run from the cloud sandbox, so jobs go through the app's
two web endpoints. Needs MODAL_KEY / MODAL_SECRET (proxy auth) in the env.

    python modal_jobs/okc.py submit --gpu h100 --timeout 3600 --cmd "bash /data/esa/run.sh" --files esa/world.py ...
    python modal_jobs/okc.py wait CALL_ID
"""
import argparse
import json
import os
import sys
import time
import urllib.request

SUBMIT = "https://ashish-soni08--okc-submit.us-east.modal.run"
RESULT = "https://ashish-soni08--okc-result.us-east.modal.run"
RATES = {"cpu": 0.2, "t4": 0.59, "h100": 3.95, "h200": 4.54}


def _req(url, data=None):
    headers = {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"],
               "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def submit(gpu, cmd, timeout_s, files):
    payload = {"gpu": gpu, "cmd": cmd, "timeout_s": timeout_s, "files": files}
    return _req(SUBMIT, payload)["call_id"]


def result(call_id):
    return _req(f"{RESULT}?call_id={call_id}")


def wait(call_id, poll=30, quiet=False):
    while True:
        r = result(call_id)
        if r.get("status") not in ("running", "pending", "queued"):
            return r
        if not quiet:
            print(".", end="", flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op")
    s = sub.add_parser("submit")
    s.add_argument("--gpu", default="cpu")
    s.add_argument("--cmd", required=True)
    s.add_argument("--timeout", type=int, default=600)
    s.add_argument("--files", nargs="*", default=[])
    s.add_argument("--prefix", default="/data/esa/")
    w = sub.add_parser("wait")
    w.add_argument("call_id")
    g = sub.add_parser("get")
    g.add_argument("call_id")
    a = ap.parse_args()
    if a.op == "submit":
        files = {a.prefix + f: open(f, encoding="utf-8").read() for f in a.files}
        cid = submit(a.gpu, a.cmd, a.timeout, files)
        print(cid)
    elif a.op in ("wait", "get"):
        r = wait(a.call_id) if a.op == "wait" else result(a.call_id)
        out = {k: v for k, v in r.items() if k not in ("stdout", "stderr")}
        print(json.dumps(out))
        if "seconds" in r:
            print(f"approx cost: see rates {RATES}")
        print("---- stdout (tail) ----\n" + str(r.get("stdout", ""))[-6000:])
        print("---- stderr (tail) ----\n" + str(r.get("stderr", ""))[-4000:], file=sys.stderr)


if __name__ == "__main__":
    main()
