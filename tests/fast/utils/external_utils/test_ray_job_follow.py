import asyncio

from ray.job_submission import JobStatus

from miles.utils.external_utils import ray_job


class _Client:
    """Replays the whole log on each tail, like Ray's file_tail_iterator."""

    def __init__(self, sessions, statuses):
        self.sessions = list(sessions)
        self.statuses = list(statuses)

    async def _tail(self, chunks, error):
        for chunk in chunks:
            yield chunk
        if error:
            raise error

    def tail_job_logs(self, submission_id):
        chunks, error = self.sessions.pop(0)
        return self._tail(chunks, error)

    def get_job_status(self, submission_id):
        return self.statuses.pop(0)


def _follow(client):
    asyncio.run(ray_job._follow_job(client, "job", asyncio.Event(), reattach_delay_s=0))


def test_follow_reattaches_when_the_stream_ends_early_without_duplicating(capsys):
    client = _Client(
        sessions=[(["ab", "cd"], None), (["abc", "def", "gh"], None)],
        statuses=[JobStatus.RUNNING, JobStatus.SUCCEEDED],
    )
    _follow(client)
    out = capsys.readouterr().out
    assert out.replace("\n[ray_job] log stream ended while job job is RUNNING; reattaching\n", "") == "abcdefgh"


def test_follow_reattaches_after_a_stream_error(capsys):
    client = _Client(
        sessions=[(["x1"], ConnectionError("dropped")), (["x1y2"], None)],
        statuses=[JobStatus.RUNNING, JobStatus.FAILED],
    )
    _follow(client)
    out = capsys.readouterr().out
    assert "log stream error" in out and out.count("x1") == 1 and out.endswith("y2")


def test_follow_returns_when_the_job_is_terminal(capsys):
    client = _Client(sessions=[(["done"], None)], statuses=[JobStatus.STOPPED])
    _follow(client)
    assert capsys.readouterr().out == "done"
