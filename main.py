from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ml import obtain_image
import io

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/generate-memory")
def generate_image_memory(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):
    image = obtain_image(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    memory_strem = io.BytesIO()
    image.save(memory_strem, format="PNG")
    memory_strem.seek(0)

    return StreamingResponse(memory_strem, media_type="image/png")
