from dataclasses import dataclass


@dataclass
class UploadedFileInput:
    name: str
    raw_bytes: bytes


@dataclass
class ProcessResult:
    logs: list[str]
    processed_files: list[dict]
    stats: dict
    pending_rows: list


@dataclass
class ScanDoneResult:
    new_added: int
    missing_files: list[str]


@dataclass
class ScriptGenerationRequest:
    db_path: str
    prompt: str
    script_style: str
    target_movie: str | None
    allow_duplicates: bool
    llm_key: str
    llm_model_name: str
    llm_base_url: str
    embedding_model: str


@dataclass
class ScriptGenerationResult:
    script: str
    material_count: int
