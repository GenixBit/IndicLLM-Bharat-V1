from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Role(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()


@dataclass
class Template:
    name: str
    system_prefix: str
    user_prefix: str
    assistant_prefix: str
    suffix: str
    system_prefix_required: bool = False


TEMPLATES: dict[str, Template] = {
    "indic_instruction": Template(
        name="indic_instruction",
        system_prefix="",
        user_prefix="<|instruction|>",
        assistant_prefix="<|response|>",
        suffix="<|endoftext|>",
    ),
    "chatml": Template(
        name="chatml",
        system_prefix="<|im_start|>system\n",
        user_prefix="<|im_start|>user\n",
        assistant_prefix="<|im_start|>assistant\n",
        suffix="<|im_end|>\n",
    ),
    "llama": Template(
        name="llama",
        system_prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        suffix="<|eot_id|>",
    ),
}


def get_template(name: str) -> Template:
    if name not in TEMPLATES:
        msg = f"Unknown template '{name}'. Available: {list(TEMPLATES.keys())}"
        raise ValueError(msg)
    return TEMPLATES[name]


def format_conversation(
    template: Template,
    messages: list[dict[str, str]],
) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            if template.system_prefix:
                parts.append(f"{template.system_prefix}{content}{template.suffix}")
        elif role == "user":
            parts.append(f"{template.user_prefix}{content}{template.suffix}")
        elif role == "assistant":
            parts.append(f"{template.assistant_prefix}{content}{template.suffix}")
    return "".join(parts)
