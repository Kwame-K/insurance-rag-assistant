from pathlib import Path
from typing import Annotated

import typer

from insurance_rag_assistant.config import (
    CHUNKS_FILE,
    DEFAULT_TOP_K,
    EVALUATION_CASES_FILE,
    EVALUATION_REPORT_FILE,
    MIN_RETRIEVAL_SCORE,
    RAW_DATA_DIR,
)
from insurance_rag_assistant.evaluation.evaluator import (
    evaluate_generation,
    evaluate_retrieval,
    load_evaluation_cases,
    write_evaluation_report,
)
from insurance_rag_assistant.generation.answer_generator import (
    GroundedAnswerGenerator,
)
from insurance_rag_assistant.ingestion.pipeline import ingest_markdown_corpus
from insurance_rag_assistant.llm.groq_client import (
    GroqStructuredLLMClient,
)
from insurance_rag_assistant.models.response import RAGResponse
from insurance_rag_assistant.models.retrieval import (
    SearchFilters,
    SearchQuery,
)
from insurance_rag_assistant.retrieval.search import (
    SemanticSearchService,
)

app = typer.Typer(
    name="insurance-rag",
    help="Grounded bilingual RAG assistant for insurance documentation.",
    no_args_is_help=True,
)

SourceDirOption = Annotated[
    Path,
    typer.Option(
        "--source-dir",
        help="Directory containing Markdown insurance documents.",
    ),
]

OutputPathOption = Annotated[
    Path,
    typer.Option(
        "--output-path",
        help="Destination JSONL file for processed chunks.",
    ),
]

RecreateOption = Annotated[
    bool,
    typer.Option(
        "--recreate",
        help="Delete and rebuild the local Qdrant collection before ingestion.",
    ),
]

QueryOption = Annotated[
    str,
    typer.Option(
        "--query",
        "-q",
        help="Insurance question or semantic search query.",
    ),
]

TopKOption = Annotated[
    int,
    typer.Option(
        "--top-k",
        min=1,
        max=20,
        help="Maximum number of retrieved passages.",
    ),
]

CoverageOption = Annotated[
    str | None,
    typer.Option("--coverage", help="Filter by coverage type."),
]

JurisdictionOption = Annotated[
    str | None,
    typer.Option("--jurisdiction", help="Filter by jurisdiction."),
]

LanguageOption = Annotated[
    str | None,
    typer.Option("--language", help="Filter by language: en or fr."),
]

DocumentTypeOption = Annotated[
    str | None,
    typer.Option("--document-type", help="Filter by document type."),
]

VersionOption = Annotated[
    str | None,
    typer.Option("--version", help="Filter by document version."),
]

EvaluationCasesOption = Annotated[
    Path,
    typer.Option(
        "--cases",
        help="Path to the retrieval evaluation cases JSON file.",
    ),
]

EvaluationOutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        help="Path to the generated JSON evaluation report.",
    ),
]


WithGenerationOption = Annotated[
    bool,
    typer.Option(
        "--with-generation",
        help=(
            "Run LLM generation evaluation. "
            "This sends one Groq request per evaluation case."
        ),
    ),
]


@app.command()
def version() -> None:
    """Display the application version."""
    typer.echo("insurance-rag 0.1.0")


@app.command()
def ingest(
    source_dir: SourceDirOption = RAW_DATA_DIR,
    output_path: OutputPathOption = CHUNKS_FILE,
    recreate: RecreateOption = False,
) -> None:
    """Ingest Markdown documents and index their embeddings locally."""
    chunks = ingest_markdown_corpus(
        source_dir=source_dir,
        output_path=output_path,
        recreate_collection=recreate,
    )

    typer.echo(f"Ingested and indexed {len(chunks)} chunks into {output_path}")


@app.command()
def search(
    query: QueryOption,
    top_k: TopKOption = DEFAULT_TOP_K,
    coverage: CoverageOption = None,
    jurisdiction: JurisdictionOption = None,
    language: LanguageOption = None,
    document_type: DocumentTypeOption = None,
    version: VersionOption = None,
) -> None:
    """Search relevant insurance document passages."""
    filters = SearchFilters.model_validate(
        {
            "coverage": coverage,
            "jurisdiction": jurisdiction,
            "language": language,
            "document_type": document_type,
            "version": version,
        }
    )

    search_query = SearchQuery(
        query=query,
        top_k=top_k,
        filters=filters,
    )

    service = SemanticSearchService(
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
    )

    try:
        result = service.search(search_query)
    finally:
        service.close()

    if not result.retrieval_sufficient:
        typer.echo(
            "No passages met the minimum relevance threshold "
            f"({MIN_RETRIEVAL_SCORE:.2f})."
        )
        raise typer.Exit(code=1)

    for item in result.results:
        location = (
            f"pages {item.page_start}-{item.page_end}"
            if item.page_start is not None and item.page_end is not None
            else "Markdown section"
        )

        typer.echo(
            f"\n[{item.rank}] Score: {item.score:.3f}\n"
            f"Document: {item.document_name}\n"
            f"Section: {item.section_title} ({location})\n"
            f"Chunk ID: {item.chunk_id}\n"
            f"Text: {item.text}\n"
        )


@app.command()
def ask(
    query: QueryOption,
    top_k: TopKOption = DEFAULT_TOP_K,
    coverage: CoverageOption = None,
    jurisdiction: JurisdictionOption = None,
    language: LanguageOption = None,
    document_type: DocumentTypeOption = None,
    version: VersionOption = None,
) -> None:
    """Answer an insurance question using cited retrieved documents."""
    filters = SearchFilters.model_validate(
        {
            "coverage": coverage,
            "jurisdiction": jurisdiction,
            "language": language,
            "document_type": document_type,
            "version": version,
        }
    )

    search_query = SearchQuery(
        query=query,
        top_k=top_k,
        filters=filters,
    )

    search_service = SemanticSearchService(
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
    )

    try:
        search_result = search_service.search(search_query)

        llm_client = GroqStructuredLLMClient()
        answer_generator = GroundedAnswerGenerator(
            llm_client=llm_client,
        )

        response = answer_generator.generate(
            question=query,
            search_result=search_result,
        )
    finally:
        search_service.close()

    _display_rag_response(response)


@app.command()
def evaluate(
    cases_path: EvaluationCasesOption = EVALUATION_CASES_FILE,
    output_path: EvaluationOutputOption = EVALUATION_REPORT_FILE,
    top_k: TopKOption = DEFAULT_TOP_K,
    with_generation: WithGenerationOption = False,
) -> None:
    """Evaluate retrieval quality and correct abstention behaviour."""
    cases = load_evaluation_cases(cases_path)
    search_service = SemanticSearchService(
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
    )

    try:
        report = evaluate_retrieval(
            search_service=search_service,
            cases=cases,
            top_k=top_k,
        )
        if with_generation:
            llm_client = GroqStructuredLLMClient()
            answer_generator = GroundedAnswerGenerator(
                llm_client=llm_client,
            )

            report["generation"] = evaluate_generation(
                search_service=search_service,
                answer_generator=answer_generator,
                cases=cases,
                top_k=top_k,
            )

    finally:
        search_service.close()

    write_evaluation_report(
        report=report,
        output_path=output_path,
    )

    recall = report["macro_recall_at_k"]
    mrr = report["mean_reciprocal_rank"]
    abstention = report["correct_abstention_rate"]

    typer.echo(f"Evaluation report written to: {output_path}")
    typer.echo(f"Cases evaluated: {report['total_cases']}")
    typer.echo(
        f"Macro Recall@{top_k}: {recall:.3f}"
        if recall is not None
        else "Macro Recall: N/A"
    )
    typer.echo(f"MRR: {mrr:.3f}" if mrr is not None else "MRR: N/A")
    typer.echo(
        f"Correct abstention rate: {abstention:.3f}"
        if abstention is not None
        else "Correct abstention rate: N/A"
    )

    generation_report = report.get("generation")

    if generation_report is not None:
        grounded_rate = generation_report["grounded_answer_rate"]
        abstention_rate = generation_report["correct_grounded_abstention_rate"]
        citation_precision = generation_report["citation_document_precision"]
        answer_constraint_rate = generation_report["answer_constraint_pass_rate"]


        typer.echo("\nGeneration evaluation:")
        typer.echo(
            f"Grounded answer rate: {grounded_rate:.3f}"
            if grounded_rate is not None
            else "Grounded answer rate: N/A"
        )
        typer.echo(
            f"Correct grounded abstention rate: {abstention_rate:.3f}"
            if abstention_rate is not None
            else "Correct grounded abstention rate: N/A"
        )
        typer.echo(
            f"Citation document precision: {citation_precision:.3f}"
            if citation_precision is not None
            else "Citation document precision: N/A"
        )
        typer.echo(
            f"Answer constraint pass rate: {answer_constraint_rate:.3f}"
            if answer_constraint_rate is not None
            else "Answer constraint pass rate: N/A"
        )
        typer.echo(f"Generation errors: {generation_report['generation_errors']}")


def _display_rag_response(response: RAGResponse) -> None:
    """Display a structured RAG response in a readable CLI format."""
    typer.echo(f"\nAnswer:\n{response.answer}")
    typer.echo(f"\nGrounded: {response.grounded}")
    typer.echo(f"Confidence: {response.confidence}")

    if response.insufficient_context_reason is not None:
        typer.echo(
            f"\nInsufficient context reason:\n{response.insufficient_context_reason}"
        )

    if response.citations:
        typer.echo("\nCitations:")

        for citation in response.citations:
            location = (
                f"pages {citation.page_start}-{citation.page_end}"
                if citation.page_start is not None and citation.page_end is not None
                else "Markdown section"
            )

            typer.echo(
                f"\n- [{citation.chunk_id}]\n"
                f"  Document: {citation.document_name}\n"
                f"  Section: {citation.section_title} ({location})\n"
                f'  Supporting quote: "{citation.quote}"'
            )
            if citation.retrieval_score is not None: typer.echo(f"  Retrieval score: {citation.retrieval_score:.3f}")
