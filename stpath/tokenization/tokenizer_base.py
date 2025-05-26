from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerBase(Protocol):
    def encode(self, *args, **kwargs):
        ...

    def decode(self, *args, **kwargs):
        ...

    @property
    def mask_token(self) -> str:
        ...

    @property
    def mask_token_id(self) -> int:
        ...

    @property
    def pad_token(self) -> str:
        ...

    @property
    def pad_token_id(self) -> int:
        ...
