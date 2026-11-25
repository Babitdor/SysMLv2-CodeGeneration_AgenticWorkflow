"""Function-calling schemas for the two pipeline stages.

The names here are the contract the client keys off: the sidebar opens on
`tool` events named `generate_sysml` / `validate_sysml`, never on markdown.
"""

GENERATE_SYSML_SCHEMA = {
    "name": "generate_sysml",
    "description": (
        "Generate a complete SysML v2 model from natural-language requirements. "
        "Returns the full model source; validate_sysml checks it afterwards."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "string",
                "description": "What the model must express, in plain language.",
            },
            "context": {
                "type": "string",
                "description": "Optional extra context: documents, existing model text, constraints.",
            },
        },
        "required": ["requirements"],
    },
}

VALIDATE_SYSML_SCHEMA = {
    "name": "validate_sysml",
    "description": (
        "Validate SysML v2 source against the official ANTLR grammar and return "
        "structured line/column diagnostics."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "SysML v2 source text to check.",
            },
        },
        "required": ["code"],
    },
}

# Served verbatim by GET /api/tools.
TOOL_MANIFEST = [GENERATE_SYSML_SCHEMA, VALIDATE_SYSML_SCHEMA]
