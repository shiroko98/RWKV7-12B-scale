from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSample:
    prompt: str
    expected_score: float
    correct_answer: int | None = None
    category: str | None = None


def get_digit_score_tokens(tokenizer) -> tuple[list[int], list[float]]:
    token_ids: list[int] = []
    score_values: list[float] = []
    for digit in range(10):
        encoded = tokenizer.encode(str(digit))
        if len(encoded) != 1:
            raise ValueError(f"Digit '{digit}' does not encode to a single token: {encoded}")
        token_ids.append(encoded[0])
        score_values.append(float(digit))
    return token_ids, score_values


MATH_SAMPLES = [
    ProbeSample(
        prompt="Estimate the result of 347 × 28. Rate your confidence from 0-9 that the answer is between 9700 and 9750.\nAnswer: ",
        expected_score=7.0,
        correct_answer=7,
        category="arithmetic",
    ),
    ProbeSample(
        prompt="Without calculating exactly, is 2^17 closer to 100000 or 150000? Rate from 0 (definitely 100000) to 9 (definitely 150000).\nAnswer: ",
        expected_score=3.0,
        correct_answer=3,
        category="powers",
    ),
    ProbeSample(
        prompt="A triangle has sides of length 5, 12, and 13. Rate from 0-9 how likely this is a right triangle.\nAnswer: ",
        expected_score=9.0,
        correct_answer=9,
        category="geometry",
    ),
    ProbeSample(
        prompt="If log₂(x) = 10, rate from 0-9 how close x is to 1000.\nAnswer: ",
        expected_score=7.0,
        correct_answer=7,
        category="logarithm",
    ),
    ProbeSample(
        prompt="Rate from 0-9 how confident you are that √(144) + √(169) > 25.\nAnswer: ",
        expected_score=1.0,
        correct_answer=1,
        category="roots",
    ),
    ProbeSample(
        prompt="A store offers 30% off, then an additional 20% off the reduced price. Rate from 0-9 how close the total discount is to 50%.\nAnswer: ",
        expected_score=4.0,
        correct_answer=4,
        category="percentage",
    ),
    ProbeSample(
        prompt="How many prime numbers are there between 1 and 20? Rate from 0 (fewer than 6) to 9 (more than 10).\nAnswer: ",
        expected_score=5.0,
        correct_answer=5,
        category="primes",
    ),
    ProbeSample(
        prompt="If you flip a fair coin 10 times, rate from 0-9 how likely you are to get exactly 5 heads.\nAnswer: ",
        expected_score=3.0,
        correct_answer=3,
        category="probability",
    ),
    ProbeSample(
        prompt="The sum of interior angles of a hexagon is ___. Rate from 0-9 how confident you are that it's 720 degrees.\nAnswer: ",
        expected_score=9.0,
        correct_answer=9,
        category="geometry",
    ),
    ProbeSample(
        prompt="Rate from 0-9: The derivative of x³ + 2x² at x=1 equals 7.\nAnswer: ",
        expected_score=9.0,
        correct_answer=9,
        category="calculus",
    ),
    ProbeSample(
        prompt="Estimate: 999 × 1001 is closest to which value? Rate from 0 (around 990000) to 9 (around 1000000).\nAnswer: ",
        expected_score=9.0,
        correct_answer=9,
        category="arithmetic",
    ),
    ProbeSample(
        prompt="Rate from 0-9: In the Fibonacci sequence (1,1,2,3,5,8,13,...), the ratio of consecutive terms approaches 1.618.\nAnswer: ",
        expected_score=9.0,
        correct_answer=9,
        category="sequences",
    ),
    ProbeSample(
        prompt="Rate from 0-9 your confidence that the integral of 1/x from 1 to e equals 1.\nAnswer: ",
        expected_score=9.0,
        correct_answer=9,
        category="calculus",
    ),
    ProbeSample(
        prompt="A group of 23 people is in a room. Rate from 0-9 how likely at least two share a birthday.\nAnswer: ",
        expected_score=5.0,
        correct_answer=5,
        category="probability",
    ),
    ProbeSample(
        prompt="Rate from 0-9: The series 1 + 1/2 + 1/4 + 1/8 + ... converges to 2.\nAnswer: ",
        expected_score=9.0,
        correct_answer=9,
        category="series",
    ),
    ProbeSample(
        prompt="Rate from 0-9 how confident you are that the number 91 is prime.\nAnswer: ",
        expected_score=1.0,
        correct_answer=1,
        category="primes",
    ),
]


EQ_SAMPLES = [
    ProbeSample(
        prompt="A friend cancels plans last minute for the third time this month. Rate from 0 (not at all frustrated) to 9 (extremely frustrated) how a typical person would feel.\nAnswer: ",
        expected_score=7.0,
        correct_answer=7,
        category="frustration",
    ),
    ProbeSample(
        prompt="Someone receives unexpected praise from a usually critical boss. Rate from 0 (suspicious) to 9 (purely happy) how they would feel.\nAnswer: ",
        expected_score=5.0,
        correct_answer=5,
        category="mixed_emotions",
    ),
    ProbeSample(
        prompt="A child shows you their drawing, clearly proud. You can see it's not technically good. Rate from 0 (be honest) to 9 (be encouraging) the socially appropriate response.\nAnswer: ",
        expected_score=8.0,
        correct_answer=8,
        category="social_judgment",
    ),
    ProbeSample(
        prompt="Two coworkers are in a heated argument. One says 'I'm fine' in a flat tone. Rate from 0 (they are fine) to 9 (they are clearly not fine) the sarcasm level.\nAnswer: ",
        expected_score=8.0,
        correct_answer=8,
        category="sarcasm_detection",
    ),
    ProbeSample(
        prompt="A person loses their job but says 'Maybe this is a blessing in disguise.' Rate from 0 (pure denial) to 9 (genuine optimism) their emotional state.\nAnswer: ",
        expected_score=4.0,
        correct_answer=4,
        category="coping",
    ),
    ProbeSample(
        prompt="At a party, someone stands alone looking at their phone. Rate from 0 (perfectly comfortable) to 9 (socially anxious) the most likely emotional state.\nAnswer: ",
        expected_score=6.0,
        correct_answer=6,
        category="social_reading",
    ),
    ProbeSample(
        prompt="A student gets a 92/100 on a test they studied weeks for. Their friend who barely studied got 95. Rate from 0 (purely happy) to 9 (envious) the student's likely feeling.\nAnswer: ",
        expected_score=6.0,
        correct_answer=6,
        category="comparison",
    ),
    ProbeSample(
        prompt="Someone apologizes by saying 'I'm sorry you feel that way.' Rate from 0 (genuine apology) to 9 (non-apology) the sincerity.\nAnswer: ",
        expected_score=8.0,
        correct_answer=8,
        category="sincerity",
    ),
    ProbeSample(
        prompt="A person laughs at something sad. Rate from 0 (inappropriate) to 9 (common coping mechanism) how normal this response is.\nAnswer: ",
        expected_score=6.0,
        correct_answer=6,
        category="coping",
    ),
    ProbeSample(
        prompt="After a breakup, someone immediately starts dating again. Rate from 0 (healthy moving on) to 9 (avoidance behavior) the psychological assessment.\nAnswer: ",
        expected_score=7.0,
        correct_answer=7,
        category="psychology",
    ),
    ProbeSample(
        prompt="A teenager rolls their eyes when asked to do chores. Rate from 0 (rebellious) to 9 (normal developmental behavior) the appropriate interpretation.\nAnswer: ",
        expected_score=7.0,
        correct_answer=7,
        category="development",
    ),
    ProbeSample(
        prompt="A manager uses the phrase 'Let's circle back on this.' Rate from 0 (genuine interest) to 9 (polite dismissal) the likely intent.\nAnswer: ",
        expected_score=7.0,
        correct_answer=7,
        category="workplace",
    ),
]


JSON_SAMPLES = [
    ProbeSample(
        prompt='Given the text: "Apache/2.4.41 (Ubuntu) Server"\nExtract the following fields as valid JSON:\n{"server": "...", "version": "...", "os": "..."}\n\nRate from 0-9 how confident you are in producing valid JSON output.\nAnswer: ',
        expected_score=8.0,
        correct_answer=8,
        category="simple_extraction",
    ),
    ProbeSample(
        prompt="Convert this key-value data to a JSON array:\nName: Alice, Age: 30\nName: Bob, Age: 25\nName: Carol, Age: 35\n\nRate from 0-9 your confidence in outputting syntactically valid JSON.\nAnswer: ",
        expected_score=8.0,
        correct_answer=8,
        category="array_format",
    ),
    ProbeSample(
        prompt="Given nested data:\nCompany: Acme Corp\n  Department: Engineering\n    Team Lead: Alice\n    Members: Bob, Carol\n  Department: Marketing\n    Team Lead: Dave\n\nProduce nested JSON. Rate from 0-9 your confidence in valid nested JSON.\nAnswer: ",
        expected_score=6.0,
        correct_answer=6,
        category="nested_structure",
    ),
    ProbeSample(
        prompt='Extract structured data from this network banner:\n<html><head><title>D-Link DCS-930L</title></head></html>\nRequired JSON format:\n{"brand": "...", "product": "...", "type": "camera"}\n\nImportant: Only extract what is explicitly in the banner.\nRate from 0-9 confidence in grounded JSON extraction.\nAnswer: ',
        expected_score=7.0,
        correct_answer=7,
        category="grounded_extraction",
    ),
    ProbeSample(
        prompt='Given this ambiguous text:\n"Welcome to our website - Powered by nginx"\n\nExtract ALL of these fields (use null if not found):\n{"brand": "...", "product": "...", "version": "...", "os": "...", "device_type": "..."}\n\nRate from 0-9 how well you can handle null/missing fields.\nAnswer: ',
        expected_score=6.0,
        correct_answer=6,
        category="null_handling",
    ),
    ProbeSample(
        prompt='Parse this multi-line server response into JSON:\n\nHTTP/1.1 200 OK\nServer: Microsoft-IIS/10.0\nX-Powered-By: ASP.NET\nContent-Type: text/html\n\n{"server": "...", "version": "...", "framework": "...", "content_type": "..."}\n\nRate from 0-9 your confidence in complete, valid JSON.\nAnswer: ',
        expected_score=8.0,
        correct_answer=8,
        category="header_parsing",
    ),
    ProbeSample(
        prompt='Text contains special characters that need JSON escaping:\nPath: C:\\Users\\admin\\Desktop\nMessage: He said "hello" and left\nTab:\there\n\nProduce valid JSON with properly escaped strings.\nRate from 0-9 confidence in correct JSON escaping.\nAnswer: ',
        expected_score=5.0,
        correct_answer=5,
        category="escaping",
    ),
    ProbeSample(
        prompt='Generate a JSON response that follows this EXACT schema:\n{\n  "results": [{"id": int, "name": str, "score": float}],\n  "total": int,\n  "page": int\n}\n\nSample data: 3 results, page 1.\nRate from 0-9 your confidence in schema-compliant JSON.\nAnswer: ',
        expected_score=7.0,
        correct_answer=7,
        category="schema_compliance",
    ),
    ProbeSample(
        prompt='CRITICAL: Output ONLY valid JSON, no markdown, no explanation.\nInput: The Raspberry Pi 4 Model B runs Debian 11\nOutput format: {"device": "...", "os": "...", "os_version": "..."}\n\nRate from 0-9 your ability to output ONLY JSON with no extra text.\nAnswer: ',
        expected_score=7.0,
        correct_answer=7,
        category="strict_output",
    ),
    ProbeSample(
        prompt='Parse this compound string into structured JSON:\n"nas-_-ds415play-_-synology"\n\nHint: Fields are separated by \'-_-\'.\n{"type": "...", "product": "...", "brand": "..."}\n\nRate from 0-9 confidence in correctly parsing compound strings.\nAnswer: ',
        expected_score=6.0,
        correct_answer=6,
        category="compound_parsing",
    ),
]


BUILTIN_PROBES: dict[str, list[ProbeSample]] = {
    "math": MATH_SAMPLES,
    "eq": EQ_SAMPLES,
    "json": JSON_SAMPLES,
}
