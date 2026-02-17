# Soundtest CLI - Design

## What it does

`soundtest` is a Rust CLI that turns `(object, message)` into **local audio**:

1. An OpenAI-compatible model produces a small plain-text **render plan**.
2. The program executes the plan using exactly one **local backend**:
   - `system` (system TTS rendered to a WAV file, then processed)
   - `procedural` (built-in synthesis for non-speaking/static objects)
3. For `system` TTS, the model may rewrite the text a bit for character, then an **effects chain** (dragon/robot/etc) is applied. **Tempo and pitch are independently controllable.**

Non-goals:
- Real recorded animal samples.
- Vendor/cloud TTS services.

## CLI

Single:

```
soundtest speak <object> <text...>
```

Batch (renders all sounds first, then mixes + plays them at the same time):

```
soundtest --speak <object> <message> [--speak <object> <message> ...]
```

Key flags:
- `--config <path>`: config file path (default: `~/.soundtest/config.toml`)
- `--base-url <url>` / `--token <token>` / `--model <name>`: override AI settings
- `--reasoning-effort <low|medium|high>`: default is `medium`
- `--backend <auto|system|procedural>`: default is `auto`
- `--volume <0..100>`: output volume (default: `100`)
- `--dry-run`: print plan + tools preview; do not play audio
- `--verbose`: print selected tools/effects and extra diagnostics (never prints token)
- `--ai-concurrency <n>` / `--tts-concurrency <n>` / `--dsp-concurrency <n>`: pipeline limits for batch mode

Stdout behavior:
- For TTS plans: prints the final `text:` that will be spoken.
- For procedural plans: prints the `proc:` token text.
- For batch runs: prints one line per item in input order: `<object>: <text/proc>`.

To see what tools are used without playing audio:

```
soundtest speak dragon "Hello" --dry-run --verbose
```

To render and play many voices at once:

```
soundtest --speak dog "How is the weather today?" --speak cow "What's your name?"
```

## Configuration

Default location: `~/.soundtest/config.toml`

Example:

```toml
base_url = "https://api.openai.com/v1"
token = "..."
model = "gpt-5.2"
model_reasoning_effort = "medium" # default

# Optional: OpenAI wire API selection
# - "auto" tries /responses first, then falls back to /chat/completions
wire_api = "auto" # auto|responses|chat_completions

# Optional: system TTS tool override
# - Windows default: "powershell"
# - macOS default: "say"
system_tts_binary = "powershell"
```

Credential/config fallback order:
1. CLI flags (`--base-url`, `--token`, `--model`, `--reasoning-effort`)
2. `~/.soundtest/config.toml`
3. Env vars `OPENAI_BASE_URL` / `OPENAI_API_KEY`
4. Codex CLI defaults in `~/.codex/config.toml` + `~/.codex/auth.json`

## Backend availability detection

At runtime the program detects:
- `system`: available on Windows and macOS if `system_tts_binary` resolves to an executable
  - Windows: uses `powershell.exe` to drive SAPI and render a WAV file
  - macOS: uses `say` to render a WAV file
- `procedural`: always available

The detected list is embedded into the AI prompt so `auto` selection chooses only usable backends.

## AI API (OpenAI-compatible)

Depending on `wire_api`, the client uses:
- `POST {base_url}/responses`
- `POST {base_url}/chat/completions`

`wire_api=auto` tries `/responses` first and falls back to `/chat/completions`.

`model_reasoning_effort` is sent when supported; on 400 errors the client retries once without it.

## Render plan format (model output)

The model must return ONLY plain text. No markdown. No explanation.

### System TTS plan

```
backend: system
text: <text to speak>                       (required)
preset: <neutral|dragon|robot|fairy|giant|ghost|radio>
amount: <0.0-1.0>
speed: <0.4-1.8>                             (optional; 1.0 normal)
pitch_semitones: <-24..24>                   (optional; negative lowers voice)
bass_db: <-12..18>                           (optional)
treble_db: <-12..18>                         (optional)
reverb: <0.0-1.0>                             (optional)
distortion: <0.0-1.0>                         (optional)
```

### Procedural plan

```
backend: procedural
proc: <token text for procedural synthesis>  (required)
```

Rules:
- Choose EXACTLY ONE backend.
- If `backend: procedural`, the program ignores any TTS/effect fields.
- If `backend: system`, the program ignores any `proc:` fields.
- `text:` should preserve the user's meaning and language, but may be lightly rewritten to match the object's voice.

## Effects engine

For `system` TTS, the audio pipeline is:

1. Run system TTS to render a temporary WAV file.
2. Decode WAV into mono `f32` samples.
3. Apply an effects chain:
   - **Pitch shift** (semitones) via linear resampling.
   - **Tempo (speed)** via a WSOLA-style time-stretcher.
   - **EQ** via low-shelf (bass) and high-shelf (treble) biquads.
   - **Reverb** via a lightweight comb/allpass network.
   - **Distortion** via a soft clip stage.
   - **Limiter** to prevent clipping.
4. Play audio via `rodio`.

Pitch and tempo independence:
- Pitch shifting changes duration by the pitch factor.
- A compensating time-stretch is applied so the final duration matches `speed` while pitch matches `pitch_semitones`.

## Observability (CLI + logs)

CLI:
- `--dry-run --verbose` prints the render plan and resolved local tool (`powershell.exe`/`say`) and final effect parameters.
- `--verbose` also prints per-segment details during playback.

Logs:
- JSONL at `~/.soundtest/soundtest.log.jsonl`
- Includes: AI timings, model + effort, wire API used, selected tools, effect parameters, and per-segment durations.
- The API token is never printed or logged.

## Testing

- Unit tests cover render plan parsing and the (currently unused) sanitizer module.
- `cargo test` is fully offline and does not call the AI API.
