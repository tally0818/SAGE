def reward(txt: str, gt: str) -> int:
    try:
        from math_verify import parse, verify

        txt_parsed = parse(txt)
        gt_parsed = parse(gt)
        return int(verify(gt_parsed, txt_parsed))
    except Exception:
        return 0


def math_verify_reward(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [reward(responses[i], answer[i]) for i in range(len(responses))]
