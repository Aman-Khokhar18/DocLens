import json
import os
from typing import Dict, Any

from openai import OpenAI

OPENAI_BASE_URL = "http://localhost:8000/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
OPENAI_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"

client = OpenAI(
    base_url=OPENAI_BASE_URL,  
    api_key=OPENAI_API_KEY,
)

# --- Prompt----------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledge graph extraction assistant specialized in technical documentation.
You read documentation for HTTP/RPC APIs, SDKs, libraries, CLI tools, and configuration, and extract
entities (endpoints, functions, classes, modules, parameters, resources, auth, errors, etc.)
and relations between them.

You MUST respond with a single valid JSON object and nothing else.
The JSON object must have exactly two keys: "entities" and "relations".

Be conservative: only create entities and relations that are clearly implied by the text.
"""

USER_PROMPT_TEMPLATE = """From the following technical documentation section, extract entities and relations.

The text may describe HTTP/RPC APIs, SDKs, libraries, classes, methods, modules, CLI commands,
configuration options, error types, or higher-level concepts.

### ENTITIES

Return a list called "entities". Each entity must be an object with:

- "id": a stable slug-like identifier (lowercase, no spaces, use dots or dashes).
  Examples: "user-api.create-user.post-users", "http.client", "numpy.ndarray",
            "cli.install-command", "config.timeout".
- "name": the human-readable name as in the docs (e.g. "POST /users", "HttpClient", "timeout").
- "type": one of:
  "service", "api_group", "endpoint", "operation", "resource",
  "package", "module", "class", "interface", "function", "method",
  "parameter", "request_body_field", "response_field", "header",
  "config_option", "cli_command", "auth_scheme", "error_code",
  "pagination", "rate_limit", "enum_value", "webhook", "concept",
  "other".
- "description": 1–3 sentences explaining the entity based ONLY on the text.
- "extra": an object (may be empty) with type-specific metadata. Examples:
  - for endpoints: "http_method", "path".
  - for functions/methods: "signature", "returns_type".
  - for parameters/fields: "required": true/false, "in": "query"|"path"|"header"|"cookie"|"body"|"args"|"kwargs".
  - for error codes: "status_code", "error_code".
  - for CLI commands: "syntax", "examples".
  - for config options: "default_value", "env_var".
  - for pagination: "strategy": "cursor"|"page"|"offset".

### RELATIONS

Return a list called "relations". Each relation must be an object with:

- "subject": an entity "id" from the "entities" list.
- "predicate": a short label, preferably from:
  "has_endpoint", "has_operation", "belongs_to", "part_of",
  "has_parameter", "has_request_field", "has_response_field",
  "returns", "emits_error", "requires_auth", "rate_limited_by",
  "paginates_with", "uses_resource", "uses_schema",
  "calls", "uses", "configured_by", "implemented_by", "extends",
  "see_also", "related_to".
- "object": an entity "id" from the "entities" list.
- "description": 1 sentence explaining the relation.

Guidance:
- Use "belongs_to" or "part_of" for membership and containment:
  - method -> class, function -> module, module -> package, endpoint -> api_group/service.
- Use "has_parameter" when a function/method/endpoint/CLI command accepts a parameter/argument.
- Use "has_request_field" when an endpoint accepts a JSON body field.
- Use "has_response_field" for JSON fields in the response.
- Use "returns" when a function/method/endpoint returns a type or resource.
- Use "emits_error" for errors/exceptions or HTTP error codes that may be produced.
- Use "requires_auth" when a specific auth scheme is required.
- Use "configured_by" when behavior is controlled by a config option or environment variable.
- Use "calls" when one function/method/endpoint calls another.
- Use "uses" when an entity depends on or uses another entity.
- Use "paginates_with" when a pagination mechanism is documented.
- Use "rate_limited_by" for rate-limit info.
- Use "see_also" for explicit cross-references.
- Use "related_to" only when no other predicate fits.

### OUTPUT FORMAT

Output ONLY a single JSON object with this exact shape:

{{
  "entities": [ ... ],
  "relations": [ ... ]
}}

Do not include any explanation or text outside the JSON.
Extract at most 20 entities and at most 40 relations from this chunk
---

Text:
```md
{chunk_text}
```"""


# -----------------------------------------------------------------------------------


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text.strip()
    return text[start : end + 1]



def extract_kg_from_chunk(chunk_text: str) -> Dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    completion = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    content = completion.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        json_block = _extract_json_block(content)
        try:
            parsed = json.loads(json_block)
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}

    entities = parsed.get("entities", [])
    relations = parsed.get("relations", [])
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    return {"entities": entities, "relations": relations}
