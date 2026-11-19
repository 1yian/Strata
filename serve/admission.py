import heapq

SESSION_HEADERS = ("x-session-affinity", "x-session-id", "session_id")
SESSION_KEY_MAX = 200
DEFAULT_READ_TPS = 1000.0
DEFAULT_GEN_TPS = 30.0
DEFAULT_ANSWER_TOKENS = 1000.0


def session_key(headers) -> str | None:
    for name in SESSION_HEADERS:
        value = (headers.get(name) or "").strip()
        if value:
            return value[:SESSION_KEY_MAX]
    return None


def service_s(totals: dict, prompt_tokens: int) -> float:
    read_n = totals.get("prompt_tokens", 0) - totals.get("reused", 0)
    read_s = totals.get("prompt_ms", 0.0) / 1000
    read_tps = read_n / read_s if read_n > 0 and read_s > 0 else DEFAULT_READ_TPS
    gen_n = totals.get("output_tokens", 0)
    gen_s = totals.get("decode_ms", 0.0) / 1000
    gen_tps = gen_n / gen_s if gen_n > 0 and gen_s > 0 else DEFAULT_GEN_TPS
    finished = totals.get("requests", 0)
    answer = gen_n / finished if finished > 0 and gen_n > 0 else DEFAULT_ANSWER_TOKENS
    return prompt_tokens / read_tps + answer / gen_tps


def estimate_wait(slots: int, remaining: list[float], ahead: list[float]) -> float:
    free_at = sorted(remaining)[:slots]
    free_at += [0.0] * (slots - len(free_at))
    heapq.heapify(free_at)
    for need in ahead:
        heapq.heapreplace(free_at, free_at[0] + need)
    return free_at[0]
