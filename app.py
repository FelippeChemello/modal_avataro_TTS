import base64
import binascii
import io
import tempfile
from pathlib import Path

import modal

APP_NAME = "moss-tts"
MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
HF_CACHE_DIR = "/cache"
FLASH_ATTN_WHEEL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

hf_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)

def download_models() -> None:
    from huggingface_hub import snapshot_download

    for model_id in (MODEL_ID, CODEC_ID):
        print(f"Downloading {model_id}...")
        snapshot_download(model_id, cache_dir=HF_CACHE_DIR)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch==2.9.1+cu128",
        "torchaudio==2.9.1+cu128",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "torchcodec==0.8.1",
        "transformers==5.0.0",
        "accelerate>=1.10.1",
        "huggingface-hub>=0.34.0",
        "safetensors==0.6.2",
        "numpy==2.1.0",
        "orjson==3.11.4",
        "tqdm==4.67.1",
        "PyYAML==6.0.3",
        "einops==0.8.1",
        "scipy==1.16.2",
        "librosa==0.11.0",
        "soundfile==0.13.1",
        "tiktoken==0.12.0",
        "psutil",
        "packaging",
        "ninja",
        "setuptools",
        "wheel",
        'cbor2'
    )
    .uv_pip_install(FLASH_ATTN_WHEEL, extra_options="--no-deps")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "PYTHONPATH": "/opt/MOSS-TTS",
        }
    )
    .run_function(
        download_models,
        volumes={HF_CACHE_DIR: hf_cache},
        timeout=60 * 60,
    )
    .add_local_dir(
        "MOSS-TTS/moss_tts_delay",
        remote_path="/opt/MOSS-TTS/moss_tts_delay",
        copy=True,
    )
)

app = modal.App(APP_NAME)


with image.imports():
    import flash_attn
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
    from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor


@app.cls(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={HF_CACHE_DIR: hf_cache},
)
class Model:
    @modal.enter()
    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MOSS-TTS Delay requires a CUDA GPU.")

        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            raise RuntimeError(
                "FlashAttention 2 requires an Ampere-or-newer GPU; "
                f"found compute capability {major}.{minor}."
            )

        print(
            f"Loading {MODEL_ID} on {torch.cuda.get_device_name()} with "
            f"flash-attn {flash_attn.__version__}"
        )

        model_path = snapshot_download(
            MODEL_ID,
            cache_dir=HF_CACHE_DIR,
        )

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        self.device = torch.device("cuda")
        self.processor = MossTTSDelayProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(
            self.device
        )

        self.model = MossTTSDelayModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.model.eval()

        self.sample_rate = int(self.processor.model_config.sampling_rate)

    @modal.method()
    def generate(
        self,
        text: str,
        reference_audio: str | None = None,
        reference_format: str = "wav",
        language: str | None = None,
        duration_tokens: int | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 1.7,
        top_p: float = 0.8,
        top_k: int = 25,
        repetition_penalty: float = 1.0,
    ) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        reference_path: str | None = None
        try:
            if reference_audio:
                suffix = _safe_audio_suffix(reference_format)
                reference_bytes = _decode_reference_audio(reference_audio)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(reference_bytes)
                    reference_path = f.name

            user_message = self.processor.build_user_message(
                text=text,
                reference=[reference_path] if reference_path else None,
                tokens=duration_tokens,
                language=language,
            )
            batch = self.processor([[user_message]], mode="generation")

            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_new_tokens=max_new_tokens,
                    audio_temperature=temperature,
                    audio_top_p=top_p,
                    audio_top_k=top_k,
                    audio_repetition_penalty=repetition_penalty,
                )

            messages = self.processor.decode(output)
            if not messages or messages[0] is None:
                raise RuntimeError("MOSS-TTS returned no decodable audio.")

            audio = messages[0].audio_codes_list[0]
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().float().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)

            wav = io.BytesIO()
            sf.write(wav, audio, self.sample_rate, format="WAV", subtype="PCM_16")
            return wav.getvalue()
        finally:
            if reference_path:
                Path(reference_path).unlink(missing_ok=True)


def _safe_audio_suffix(audio_format: str) -> str:
    normalized = audio_format.strip().lower().lstrip(".")
    allowed = {"wav", "mp3", "m4a", "flac", "ogg", "opus", "aac"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported reference_format {audio_format!r}; "
            f"expected one of {sorted(allowed)}."
        )
    return f".{normalized}"


def _decode_reference_audio(reference_audio: str) -> bytes:
    try:
        return base64.b64decode(reference_audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("reference_audio must be valid Base64.") from exc


@app.local_entrypoint()
def main(
    text: str = "Olá mundo!",
    output: str = "moss_tts_output.wav",
    reference_audio: str = "reference.wav",
    language: str = "Portuguese",
) -> None:
    reference_base64 = None
    reference_format = "wav"
    if reference_audio:
        reference_path = Path(reference_audio)
        reference_base64 = base64.b64encode(
            reference_path.read_bytes()
        ).decode("ascii")
        reference_format = reference_path.suffix.lstrip(".") or "wav"

    wav = MossTTS().generate.remote(
        text=text,
        reference_audio=reference_base64,
        reference_format=reference_format,
        language=language or None,
    )
    Path(output).write_bytes(wav)
    print(f"Saved {output} ({len(wav):,} bytes)")
