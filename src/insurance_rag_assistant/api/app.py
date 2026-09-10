from fastapi import FastAPI

from insurance_rag_assistant.api.routes import router

app = FastAPI(
    title="Insurance Knowledge Agent API",
    version="0.1.0",
    description=(
        "Grounded insurance-document retrieval service returning "
        "verbatim documentary evidence."
    ),
)

app.include_router(router)
