use crate::cli::BackendChoice;
use crate::config::Settings;
use crate::effects::{self, EffectParams};
use crate::logging;
use crate::render_plan::{BackendKind, RenderPlan};
use anyhow::{Context, Result, anyhow};
use rodio::Source;
use rodio::buffer::SamplesBuffer;
use serde_json::json;
use std::io::{BufReader, Read, Seek};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BackendAvailability {
    pub system_tts: Option<SystemTtsAvailability>,
    pub procedural: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SystemTtsKind {
    #[cfg(windows)]
    PowerShellSapi,
    #[cfg(target_os = "macos")]
    Say,
}

#[derive(Debug, Clone)]
pub struct SystemTtsAvailability {
    pub binary: PathBuf,
    kind: SystemTtsKind,
}

impl BackendAvailability {
    pub fn detect(settings: &Settings) -> Self {
        let system_tts = detect_system_tts(settings);

        Self {
            system_tts,
            procedural: true,
        }
    }

    pub fn available_backends_for_ai(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.system_tts.is_some() {
            out.push("system".to_owned());
        }
        out.push("procedural".to_owned());
        out
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionPreview {
    pub plan_backend: BackendKind,
    pub requested_backend: BackendChoice,

    pub resolved_tts_backend: Option<String>,
    pub resolved_tts_tool: Option<String>,

    pub text: Option<String>,
    pub proc: Option<String>,
    pub effects: Option<EffectParams>,
}

#[derive(Debug, Clone)]
pub struct RenderedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl ExecutionPreview {
    pub fn format_tools(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "plan_backend={} requested_backend={}",
            format!("{:?}", self.plan_backend).to_ascii_lowercase(),
            format!("{:?}", self.requested_backend).to_ascii_lowercase()
        ));

        match (&self.resolved_tts_backend, &self.resolved_tts_tool) {
            (Some(backend), Some(tool)) => {
                out.push_str(&format!(
                    "\nresolved_tts_backend={backend}\nresolved_tts_tool={tool}"
                ));
            }
            _ => out.push_str("\nresolved_tts_backend=none"),
        }

        if self.plan_backend == BackendKind::Procedural {
            out.push_str("\nprocedural_tool=built-in");
        }

        if let Some(effects) = &self.effects {
            out.push_str(&format!("\neffects={}", effects.to_json()));
        }

        out
    }
}

pub fn preview_execution(
    settings: &Settings,
    plan: &RenderPlan,
    requested_backend: BackendChoice,
) -> Result<ExecutionPreview> {
    let availability = BackendAvailability::detect(settings);
    let allow_fallback = requested_backend == BackendChoice::Auto;
    let tts_backend = resolve_tts_backend(plan.backend, &availability, allow_fallback)?;

    let effects_params = match plan.backend {
        BackendKind::System => Some(EffectParams::from_spec(&plan.effects)),
        BackendKind::Procedural => None,
    };

    Ok(ExecutionPreview {
        plan_backend: plan.backend,
        requested_backend,
        resolved_tts_backend: tts_backend.as_ref().map(|b| tts_backend_kind(b).to_owned()),
        resolved_tts_tool: tts_backend.as_ref().map(tts_backend_tool),
        text: plan.text.clone(),
        proc: plan.proc.clone(),
        effects: effects_params,
    })
}

pub fn synthesize_system_tts_mono(
    availability: &BackendAvailability,
    requested_backend: BackendChoice,
    text: &str,
) -> Result<RenderedAudio> {
    let allow_fallback = requested_backend == BackendChoice::Auto;
    let tts_backend = resolve_tts_backend(BackendKind::System, availability, allow_fallback)?
        .ok_or_else(|| anyhow!("system TTS backend is not available"))?;
    let (samples, sample_rate) = synthesize_tts_mono(&tts_backend, text)?;
    Ok(RenderedAudio { samples, sample_rate })
}

pub async fn execute_render_plan(
    settings: &Settings,
    plan: &RenderPlan,
    requested_backend: BackendChoice,
    verbose: bool,
    volume: f32,
) -> Result<()> {
    let start = Instant::now();
    let availability = BackendAvailability::detect(settings);
    let audio = AudioOut::new(volume).context("failed to initialize audio output")?;

    let allow_fallback = requested_backend == BackendChoice::Auto;
    let tts_backend = resolve_tts_backend(plan.backend, &availability, allow_fallback)?;
    let effects_params = match plan.backend {
        BackendKind::System => Some(EffectParams::from_spec(&plan.effects)),
        BackendKind::Procedural => None,
    };

    logging::info(
        "backend.start",
        json!({
            "plan_backend": format!("{:?}", plan.backend).to_ascii_lowercase(),
            "requested_backend": format!("{:?}", requested_backend).to_ascii_lowercase(),
            "resolved_tts_backend": tts_backend.as_ref().map(tts_backend_kind),
            "resolved_tts_tool": tts_backend.as_ref().map(tts_backend_tool),
            "available_backends": availability.available_backends_for_ai(),
            "volume": volume,
            "effects_spec": {
                "preset": &plan.effects.preset,
                "amount": plan.effects.amount,
                "speed": plan.effects.speed,
                "pitch_semitones": plan.effects.pitch_semitones,
                "bass_db": plan.effects.bass_db,
                "treble_db": plan.effects.treble_db,
                "reverb": plan.effects.reverb,
                "distortion": plan.effects.distortion,
            },
            "effects_params": effects_params.as_ref().map(|p| p.to_json()),
        }),
    );

    if verbose {
        eprintln!(
            "tools: {}",
            preview_execution(settings, plan, requested_backend)?.format_tools()
        );
    }

    match plan.backend {
        BackendKind::Procedural => {
            let tokens = plan
                .proc
                .as_deref()
                .ok_or_else(|| anyhow!("render plan missing `proc:` for procedural backend"))?;
            println!("{tokens}");

            let segment_start = Instant::now();
            logging::info(
                "backend.segment.start",
                json!({
                    "segment_index": 0,
                    "kind": "procedural",
                    "proc_chars": tokens.chars().count(),
                }),
            );
            if verbose {
                eprintln!("segment 0: procedural (built-in)");
            }

            let result = crate::procedural::play_token_text(&audio, tokens);
            if let Err(err) = result {
                logging::error(
                    "backend.segment.end",
                    json!({
                        "segment_index": 0,
                        "kind": "procedural",
                        "status": "error",
                        "duration_ms": segment_start.elapsed().as_millis(),
                        "error": format!("{err:#}"),
                    }),
                );
                logging::error(
                    "backend.end",
                    json!({
                        "status": "error",
                        "duration_ms": start.elapsed().as_millis(),
                        "error": format!("{err:#}"),
                    }),
                );
                return Err(err);
            }

            logging::info(
                "backend.segment.end",
                json!({
                    "segment_index": 0,
                    "kind": "procedural",
                    "status": "ok",
                    "duration_ms": segment_start.elapsed().as_millis(),
                }),
            );
        }
        BackendKind::System => {
            let text = plan
                .text
                .as_deref()
                .ok_or_else(|| anyhow!("render plan missing `text:` for TTS backend"))?;
            println!("{text}");

            let Some(tts_backend) = tts_backend else {
                let err =
                    anyhow!("render plan requested TTS backend but no TTS backend is available");
                logging::error(
                    "backend.end",
                    json!({
                        "status": "error",
                        "duration_ms": start.elapsed().as_millis(),
                        "error": format!("{err:#}"),
                    }),
                );
                return Err(err);
            };

            let effects_params = effects_params.unwrap_or_else(EffectParams::neutral);

            let segment_start = Instant::now();
            logging::info(
                "backend.segment.start",
                json!({
                    "segment_index": 0,
                    "kind": "tts",
                    "tts_backend": tts_backend_kind(&tts_backend),
                    "tts_tool": tts_backend_tool(&tts_backend),
                    "text_chars": text.chars().count(),
                    "effects_params": effects_params.to_json(),
                }),
            );

            if verbose {
                eprintln!(
                    "segment 0: tts via {} (tool={})",
                    tts_backend_kind(&tts_backend),
                    tts_backend_tool(&tts_backend)
                );
                eprintln!("segment 0: effects {}", effects_params.to_json());
            }

            let tts_start = Instant::now();
            let (samples, sample_rate) = synthesize_tts_mono(&tts_backend, text)?;
            let tts_ms = tts_start.elapsed().as_millis();

            let fx_start = Instant::now();
            let processed = effects::apply_effects_mono(&samples, sample_rate, &effects_params);
            let fx_ms = fx_start.elapsed().as_millis();

            let play_start = Instant::now();
            let source = SamplesBuffer::new(1, sample_rate, processed);
            audio.play(source)?;
            let play_ms = play_start.elapsed().as_millis();

            logging::info(
                "backend.segment.end",
                json!({
                    "segment_index": 0,
                    "kind": "tts",
                    "status": "ok",
                    "duration_ms": segment_start.elapsed().as_millis(),
                    "tts_ms": tts_ms,
                    "effects_ms": fx_ms,
                    "play_ms": play_ms,
                }),
            );
        }
    }

    logging::info(
        "backend.end",
        json!({
            "status": "ok",
            "duration_ms": start.elapsed().as_millis(),
        }),
    );

    Ok(())
}

#[derive(Debug, Clone)]
enum TtsBackend {
    System(SystemTtsAvailability),
}

fn tts_backend_kind(backend: &TtsBackend) -> &'static str {
    match backend {
        TtsBackend::System(_) => "system",
    }
}

fn tts_backend_tool(backend: &TtsBackend) -> String {
    match backend {
        TtsBackend::System(s) => s.binary.to_string_lossy().to_string(),
    }
}

fn resolve_tts_backend(
    plan_backend: BackendKind,
    availability: &BackendAvailability,
    allow_fallback: bool,
) -> Result<Option<TtsBackend>> {
    if plan_backend == BackendKind::Procedural {
        return Ok(None);
    }

    let Some(system_tts) = availability.system_tts.clone() else {
        if allow_fallback {
            return Err(anyhow!(
                "system TTS backend is not available on this platform; use `--backend procedural`"
            ));
        }
        return Err(anyhow!(
            "system TTS backend requested but not available (set system_tts_binary or use procedural)"
        ));
    };

    Ok(Some(TtsBackend::System(system_tts)))
}

fn synthesize_tts_mono(backend: &TtsBackend, text: &str) -> Result<(Vec<f32>, u32)> {
    match backend {
        TtsBackend::System(system) => synthesize_with_system_tts(system, text),
    }
}

fn resolve_executable(spec: &str) -> Option<PathBuf> {
    let path = PathBuf::from(spec);
    if path.exists() {
        return Some(path);
    }
    which::which(spec).ok()
}

fn detect_system_tts(settings: &Settings) -> Option<SystemTtsAvailability> {
    let binary = resolve_executable(&settings.system_tts_binary)?;

    #[cfg(windows)]
    {
        return Some(SystemTtsAvailability {
            binary,
            kind: SystemTtsKind::PowerShellSapi,
        });
    }

    #[cfg(target_os = "macos")]
    {
        return Some(SystemTtsAvailability {
            binary,
            kind: SystemTtsKind::Say,
        });
    }

    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = binary;
        None
    }
}

fn synthesize_with_system_tts(
    system: &SystemTtsAvailability,
    text: &str,
) -> Result<(Vec<f32>, u32)> {
    match system.kind {
        #[cfg(windows)]
        SystemTtsKind::PowerShellSapi => synthesize_windows_sapi_powershell(&system.binary, text),
        #[cfg(target_os = "macos")]
        SystemTtsKind::Say => synthesize_macos_say(&system.binary, text),
    }
}

#[cfg(windows)]
fn synthesize_windows_sapi_powershell(powershell: &PathBuf, text: &str) -> Result<(Vec<f32>, u32)> {
    let dir = tempfile::tempdir()?;
    let wav_path = dir.path().join("soundtest_system.wav");
    let text_path = dir.path().join("soundtest_text.txt");
    std::fs::write(&text_path, text)?;

    let script = r#"
$ErrorActionPreference = 'Stop'
$wavPath  = $env:SOUNDTEST_WAV_PATH
$textPath = $env:SOUNDTEST_TEXT_PATH
if (-not $wavPath)  { throw "missing env SOUNDTEST_WAV_PATH" }
if (-not $textPath) { throw "missing env SOUNDTEST_TEXT_PATH" }
$text = Get-Content -LiteralPath $textPath -Raw
$voice  = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream
$stream.Open($wavPath, 3, $true)
$voice.AudioOutputStream = $stream
$null = $voice.Speak($text)
$stream.Close()
"#;

    let output = Command::new(powershell)
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-Command")
        .arg(script)
        .env("SOUNDTEST_WAV_PATH", &wav_path)
        .env("SOUNDTEST_TEXT_PATH", &text_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to run {}", powershell.to_string_lossy()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "system TTS (SAPI via PowerShell) failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }

    let file = std::fs::File::open(&wav_path)?;
    decode_audio_to_mono_f32(file)
}

#[cfg(target_os = "macos")]
fn synthesize_macos_say(say: &PathBuf, text: &str) -> Result<(Vec<f32>, u32)> {
    let dir = tempfile::tempdir()?;
    let wav_path = dir.path().join("soundtest_system.wav");
    let text_path = dir.path().join("soundtest_text.txt");
    std::fs::write(&text_path, text)?;

    let output = Command::new(say)
        .arg("-f")
        .arg(&text_path)
        .arg("-o")
        .arg(&wav_path)
        .arg("--data-format=LEI16@22050")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to run {}", say.to_string_lossy()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "system TTS (say) failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr.trim()
        ));
    }

    let file = std::fs::File::open(&wav_path)?;
    decode_audio_to_mono_f32(file)
}

fn decode_audio_to_mono_f32<R>(reader: R) -> Result<(Vec<f32>, u32)>
where
    R: Read + Seek + Send + Sync + 'static,
{
    let decoder = rodio::Decoder::new(BufReader::new(reader))?;
    let sample_rate = decoder.sample_rate();
    let channels = decoder.channels() as usize;
    let samples: Vec<f32> = decoder.convert_samples().collect();

    if channels <= 1 {
        return Ok((samples, sample_rate));
    }

    let frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += samples[frame * channels + ch];
        }
        mono.push(sum / channels as f32);
    }

    Ok((mono, sample_rate))
}

pub struct AudioOut {
    _stream: rodio::OutputStream,
    handle: rodio::OutputStreamHandle,
    volume: f32,
}

impl AudioOut {
    pub fn new(volume: f32) -> Result<Self> {
        let (_stream, handle) = rodio::OutputStream::try_default()?;
        Ok(Self {
            _stream,
            handle,
            volume: volume.clamp(0.0, 1.0),
        })
    }

    pub fn play<S>(&self, source: S) -> Result<()>
    where
        S: Source + Send + 'static,
        S::Item: rodio::Sample + Send,
        f32: rodio::cpal::FromSample<S::Item>,
    {
        let sink = rodio::Sink::try_new(&self.handle)?;
        sink.set_volume(self.volume);
        sink.append(source);
        sink.sleep_until_end();
        Ok(())
    }
}
