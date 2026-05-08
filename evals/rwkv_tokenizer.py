from __future__ import annotations


class RWKVTokenizer:
    def __init__(self, file_name: str):
        self.idx2token: dict[int, bytes] = {}
        sorted_tokens: list[bytes] = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for line in lines:
            idx = int(line[: line.index(" ")])
            token = eval(line[line.index(" ") : line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            if not isinstance(token, bytes):
                raise TypeError("Tokenizer entry is not bytes.")
            sorted_tokens.append(token)
            self.idx2token[idx] = token

        self.token2idx = {value: key for key, value in self.idx2token.items()}
        self.unknown_token = b"\xef\xbf\xbd"
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for token in reversed(sorted_tokens):
            if len(token) < 2:
                continue
            s0 = int(token[0])
            s1 = int(token[1])
            self.table[s0][s1].append(token)
            self.wlen[s0] = max(self.wlen[s0], len(token))
            self.good[s0].add(s1)

    def encode_bytes(self, src: bytes) -> list[int]:
        src_len = len(src)
        tokens: list[int] = []
        idx = 0
        while idx < src_len:
            current = src[idx : idx + 1]
            if idx < src_len - 1:
                s0 = int(src[idx])
                s1 = int(src[idx + 1])
                if s1 in self.good[s0]:
                    candidate = src[idx : idx + self.wlen[s0]]
                    try:
                        current = next(filter(candidate.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[current])
            idx += len(current)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(self.idx2token.get(token, self.unknown_token) for token in tokens)

    def encode(self, src: str) -> list[int]:
        return self.encode_bytes(src.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def count_unknown(self, tokens: list[int]) -> int:
        return sum(1 for token in tokens if token not in self.idx2token)
