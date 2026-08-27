# Render (and any Procfile-aware host) reads this. Keep it in the repo so the
# start command is version-controlled rather than living only in a dashboard
# field nobody can review.
#
# --workers 1 IS LOAD-BEARING, not a performance oversight.
#
# Exact sampling keeps state in process memory: the background job store
# (_jobs), the per-client session store (_sessions), and the transfer-matrix
# cache (sixvertex/exact.py). None of it is shared between processes. With two
# or more workers a load balancer will happily start a job on worker 1 and
# route the client's status poll to worker 2, which has never heard of it.
# Verified directly: polling a live job on the other worker returns
# HTTP 404 "unknown or expired job", and a session created on one worker is
# rejected by the other.
#
# If you need real concurrency, the fix is to move that state out of process
# (Redis or a database) -- not to raise this number.
#
# --timeout 120 only guards against slow cold starts; every request is short
# by design, because long computations run as background jobs and the client
# polls for them.
web: gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
