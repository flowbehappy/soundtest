use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

fn default_dsp_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[derive(Debug, Parser)]
#[command(
    name = "soundtest",
    version,
    about = "AI-driven object-voice TTS and procedural sounds"
)]
pub struct Cli {
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    #[arg(long, global = true)]
    pub base_url: Option<String>,

    #[arg(long, global = true)]
    pub token: Option<String>,

    #[arg(long, global = true)]
    pub model: Option<String>,

    #[arg(long, global = true)]
    pub reasoning_effort: Option<String>,

    /// Batch mode: repeat `--speak <OBJECT> <MESSAGE>` to enqueue multiple sounds.
    #[arg(long, value_names = ["OBJECT", "MESSAGE"], num_args = 2)]
    pub speak: Vec<String>,

    #[arg(long, value_enum, default_value_t = BackendChoice::Auto, global = true)]
    pub backend: BackendChoice,

    #[arg(long, global = true)]
    pub dry_run: bool,

    #[arg(long, global = true)]
    pub verbose: bool,

    /// Output volume (0-100).
    #[arg(
        long,
        default_value_t = 100,
        global = true,
        value_parser = clap::value_parser!(u8).range(0..=100)
    )]
    pub volume: u8,

    /// Max in-flight AI planning calls when batching.
    #[arg(long, default_value_t = 8, global = true)]
    pub ai_concurrency: usize,

    /// Max in-flight system TTS processes when batching.
    #[arg(long, default_value_t = 2, global = true)]
    pub tts_concurrency: usize,

    /// Max in-flight DSP/procedural rendering jobs when batching.
    #[arg(long, default_value_t = default_dsp_concurrency(), global = true)]
    pub dsp_concurrency: usize,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Speak(SpeakArgs),
}

#[derive(Debug, Args)]
pub struct SpeakArgs {
    pub object: String,
    pub text: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendChoice {
    Auto,
    System,
    Procedural,
}
