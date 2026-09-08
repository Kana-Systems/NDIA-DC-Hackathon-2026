"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secrets are never included in model representations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "Government Contract Review Demo"
    workspace_username: str = Field(
        default="judge",
        validation_alias=AliasChoices("WORKSPACE_USERNAME", "GRADIO_USERNAME"),
    )
    workspace_password: SecretStr = Field(
        min_length=12,
        validation_alias=AliasChoices("WORKSPACE_PASSWORD", "GRADIO_PASSWORD"),
    )
    max_upload_mb: int = Field(default=15, ge=1, le=100)
    max_docx_members: int = Field(default=2_000, ge=10, le=20_000)
    max_docx_uncompressed_mb: int = Field(default=50, ge=1, le=500)
    max_docx_compression_ratio: int = Field(default=100, ge=10, le=1_000)
    max_pdf_pages: int = Field(default=500, ge=1, le=5_000)
    max_extracted_characters: int = Field(default=2_000_000, ge=1_000)
    max_document_segments: int = Field(default=20_000, ge=10)
    max_review_concurrency: int = Field(default=2, ge=1, le=32)
    parse_timeout_seconds: float = Field(default=20, ge=0.1, le=300)
    parse_memory_limit_mb: int = Field(default=768, ge=128, le=8_192)
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx")
    classifier_enabled: bool = True
    classifier_model_dir: str = "/srv/app/ml/model"
    classifier_domain_mapping_path: str = "ml/cuad_category_domain_mapping.json"
    classifier_threshold: float = Field(default=0.5, ge=0, le=1)
    classifier_thresholds_path: str = ""
    bedrock_enabled: bool = False
    model_review_enabled: bool = False
    model_review_max_candidates: int = Field(default=12, ge=1, le=30)
    local_corpus_path: str = ""
    model_selection_path: str = ""
    bedrock_model_id: str = "openai.gpt-5.6-terra"
    aws_region: str = "us-gov-west-1"
    bedrock_timeout_seconds: int = Field(default=25, ge=1, le=120)
    opensearch_endpoint: str | None = None
    opensearch_index: str = "government-contract-knowledge-v1"
    opensearch_vector_enabled: bool = False
    opensearch_top_k: int = Field(default=8, ge=1, le=50)
    titan_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    rules_path: str = "knowledge/rules/review_rules.yaml"
    security_domain: str = Field(default="demo", pattern=r"^[a-z][a-z0-9-]{1,19}$")
    identity_mode: str = Field(default="demo", pattern=r"^(demo|oidc)$")
    demo_identity_enabled: bool = True
    demo_jwt_secret: SecretStr = Field(min_length=32)
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_algorithms: tuple[str, ...] = ("RS256",)
    enterprise_index: str = "j2-enterprise-intelligence-v1"
    max_rag_evidence: int = Field(default=8, ge=1, le=25)
    document_registry_table: str = ""
    entity_registry_table: str = ""
    change_event_table: str = ""
    source_bucket: str = ""
    graph_connector_secret_arn: str = ""

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(
                extension.strip().lower() for extension in value.split(",") if extension.strip()
            )
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
