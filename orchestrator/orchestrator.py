from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()


class WorkflowInput(BaseModel):
    text: str


@app.post("/run/summarize-research")
async def run_workflow(payload: WorkflowInput):
    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Call summarizer agent
            summarizer_resp = await client.post(
                "http://summarizer:8000/run/summarizer",
                json={"text": payload.text}
            )
            summarizer_data = summarizer_resp.json()
            summary = summarizer_data["summary"]

            # Step 2: Call researcher agent with summary
            researcher_resp = await client.post(
                "http://researcher:8000/run/researcher",
                json={"question": summary}
            )
            researcher_data = researcher_resp.json()

        return {
            "summary": summary,
            "research_result": researcher_data["result"],
            "_trace": {
                "summarizer": summarizer_data,
                "researcher": researcher_data
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))