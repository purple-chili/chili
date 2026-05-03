mod completer;
mod logger;
mod pipe;
mod validator;

use crate::logger::LOG_FN;
use crate::pipe::Pipe;
use crate::validator::ChiliValidator;
use chili_core::EngineState;
use chili_op::BUILT_IN_FN;
use clap::Parser;
use env_logger::Target;
use home::home_dir;
use log::{debug, error, info};

use crate::completer::ChiliCompleter;
use nu_ansi_term::{Color, Style};
use reedline::{
    ColumnarMenu, DefaultHinter, DefaultPrompt, DefaultPromptSegment, Emacs, ExternalPrinter,
    FileBackedHistory, KeyCode, KeyModifiers, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu,
    Signal, default_emacs_keybindings,
};
use std::fs::File;
use std::io::IsTerminal;

use std::path::PathBuf;
use std::process::exit;
use std::str::FromStr;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about = "an implementation of runtime for chili language", long_about = None, name = "chili")]
struct Args {
    /// Optional fixed source file path
    #[arg(index = 1)]
    src: Option<String>,

    // Optional port number for IPC
    #[arg(short, long, default_value_t = 0)]
    port: i32,

    // Optional flag for IPC to allow remote connections
    #[arg(short, long, default_value = "false")]
    remote: bool,

    // Optional list of users for authentication
    #[arg(short, long, value_delimiter = ' ', num_args = 1..)]
    users: Vec<String>,

    // Optional log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    // Optional directory for outputting logs
    #[arg(short = 'd', long = "dir", default_value = "")]
    log_dir: String,

    // Optional interval for executing jobs scheduler
    #[arg(short, long, default_value = "0")]
    interval: u64,

    // Optional keyword arguments
    #[arg(short, long, default_value = "")]
    kwargs: String,

    /// Optional memory limit in GB (default: 0 for unlimited, at least 1 GB)
    #[arg(short = 'm', long = "memory", default_value_t = 0.0)]
    memory_limit: f64,

    /// Optional flag to enable lazy evaluation
    #[arg(short = 'L', long, default_value = "false")]
    lazy: bool,

    /// Optional flag to enable pepper syntax
    #[arg(short = 'P', long, default_value = "false")]
    pepper: bool,

    /// Skip the interactive REPL; run as headless daemon (auto-detected when stdin is not a TTY with --port)
    #[arg(long, default_value = "false")]
    headless: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let log_level = log::LevelFilter::from_str(&args.log_level.to_lowercase())
        .unwrap_or(log::LevelFilter::Info);

    let debug = log_level == log::LevelFilter::Debug;

    // Headless mode: explicit flag, or port requested with non-TTY stdin (daemon launch)
    let is_headless = args.headless || (args.port > 0 && !std::io::stdin().is_terminal());

    // ExternalPrinter is only needed for the interactive REPL; skip in headless mode
    let printer: Option<ExternalPrinter<String>> = if is_headless || !args.log_dir.is_empty() {
        None
    } else {
        Some(ExternalPrinter::default())
    };

    let target = if !args.log_dir.is_empty() {
        let log_dir = PathBuf::from(&args.log_dir);
        if !log_dir.exists()
            && let Err(e) = std::fs::create_dir_all(&log_dir)
        {
            eprintln!("failed to create log directory: {}", e);
            exit(1);
        }
        let log_file = log_dir.join(format!(
            "chili_{}.log",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        ));
        match File::create(log_file) {
            Ok(file) => Target::Pipe(Box::new(file)),
            Err(e) => {
                eprintln!("failed to create log file: {}", e);
                exit(1);
            }
        }
    } else if let Some(ref p) = printer {
        let pipe = Pipe::new(p.clone());
        Target::Pipe(Box::new(pipe))
    } else {
        // headless: log to stderr (works without a TTY)
        Target::Stderr
    };

    env_logger::Builder::new()
        .filter_level(log_level)
        .format_module_path(false)
        .format_target(false)
        .format_timestamp_millis()
        .target(target)
        .init();

    if args.pepper {
        println!(
            "\x1b[1;32m\
            *                                                             \n\
            *   ████████   ██████  ████████  ████████   ██████  ████████  \n\
            *  ▒▒███▒▒███ ███▒▒███▒▒███▒▒███▒▒███▒▒███ ███▒▒███▒▒███▒▒███ \n\
            *   ▒███ ▒███▒███████  ▒███ ▒███ ▒███ ▒███▒███████  ▒███ ▒▒▒  \n\
            *   ▒███ ▒███▒███▒▒▒   ▒███ ▒███ ▒███ ▒███▒███▒▒▒   ▒███      \n\
            *   ▒███████ ▒▒██████  ▒███████  ▒███████ ▒▒██████  █████     \n\
            *   ▒███▒▒▒   ▒▒▒▒▒▒   ▒███▒▒▒   ▒███▒▒▒   ▒▒▒▒▒▒  ▒▒▒▒▒      \n\
            *   ▒███               ▒███      ▒███                         \n\
            *   █████              █████     █████                        \n\
            *  ▒▒▒▒▒              ▒▒▒▒▒     ▒▒▒▒▒                         \n\
            *                                                             \n\
            *                                                 ver {:>6 }  \n\
            *                                                 pid {:>6 }  \n\
            *                                                             \x1b[0m",
            env!("CARGO_PKG_VERSION"),
            std::process::id()
        );
    } else {
        println!(
            "\x1b[1;32m\
            *                                         \n\
            *            █████       ███  ████   ███  \n\
            *           ▒▒███       ▒▒▒  ▒▒███  ▒▒▒   \n\
            *    ██████  ▒███████   ████  ▒███  ████  \n\
            *   ███▒▒███ ▒███▒▒███ ▒▒███  ▒███ ▒▒███  \n\
            *  ▒███ ▒▒▒  ▒███ ▒███  ▒███  ▒███  ▒███  \n\
            *  ▒███  ███ ▒███ ▒███  ▒███  ▒███  ▒███  \n\
            *  ▒▒██████  ████ █████ █████ █████ █████ \n\
            *   ▒▒▒▒▒▒  ▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒  \n\
            *                                         \n\
            *                             ver {:>6 }  \n\
            *                             pid {:>6 }  \n\
            *                                         \x1b[0m",
            env!("CARGO_PKG_VERSION"),
            std::process::id()
        );
    }

    unsafe { std::env::set_var("CHILI_SYNTAX", if args.pepper { "pepper" } else { "chili" }) };

    let mut state = EngineState::new(debug, args.lazy, args.pepper);

    if args.memory_limit > 0.0 {
        state.set_memory_limit(args.memory_limit);
    }

    if args.interval > 0 {
        state.set_interval(args.interval);
    }

    if debug {
        info!("Debug mode is enabled");
    }

    if args.lazy {
        unsafe { std::env::set_var("CHILI_LAZY_MODE", "true") };
        info!("Lazy evaluation mode is enabled");
    }

    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);

    debug!("args: {:?}", args);

    if !args.kwargs.is_empty() {
        match state.parse("", &args.kwargs) {
            Ok(nodes) => match state.eval_ast(nodes, "", &args.kwargs) {
                Ok(any) => state.set_var("kwargs", any).unwrap(),
                Err(e) => {
                    eprintln!("\x1b[1;91m{}\x1b[0m", e);
                    exit(1);
                }
            },
            Err(e) => {
                eprintln!("\x1b[1;91m{}\x1b[0m", e);
                exit(1);
            }
        }
    }

    let arc_state = Arc::new(state);
    arc_state.set_arc_self(Arc::clone(&arc_state)).unwrap();

    if args.port > 0 {
        let state_tcp = Arc::clone(&arc_state);
        let port = args.port;
        let remote = args.remote;
        let users = args.users.clone();
        thread::spawn(move || {
            state_tcp.start_tcp_listener(port, remote, users);
        });
    }

    if args.src.is_some()
        && let Err(e) = arc_state.import_source_path("", &args.src.unwrap())
    {
        eprintln!("\x1b[1;91m{}\x1b[0m", e);
        if !debug {
            exit(1);
        }
    }

    arc_state.start_job_scheduler();
    arc_state.start_memory_monitor();

    if is_headless {
        info!(
            "headless mode: REPL disabled, IPC server active on port {}",
            args.port
        );
        // Park the main thread indefinitely; the IPC server thread handles all traffic.
        // A process supervisor (systemd, supervisord) can send SIGTERM to stop.
        // Loop to guard against spurious wakeups from thread::park().
        loop {
            std::thread::park();
        }
    } else {
        let completer = ChiliCompleter::new(&arc_state);

        let prompt_prefix = if args.pepper {
            "p ".to_owned()
        } else {
            "c ".to_owned()
        };
        let prompt = DefaultPrompt::new(
            DefaultPromptSegment::Basic(prompt_prefix),
            DefaultPromptSegment::Empty,
        );

        let history_path = if args.pepper {
            home_dir().unwrap().join(".pepper_history")
        } else {
            home_dir().unwrap().join(".chili_history")
        };

        let history = Box::new(
            FileBackedHistory::with_file(1000, history_path)
                .expect("Failed to create history file"),
        );
        let hinter = DefaultHinter::default().with_style(Style::new().italic().fg(Color::DarkGray));

        let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));

        let mut keybindings = default_emacs_keybindings();
        keybindings.add_binding(
            KeyModifiers::NONE,
            KeyCode::Tab,
            ReedlineEvent::UntilFound(vec![
                ReedlineEvent::Menu("completion_menu".to_string()),
                ReedlineEvent::MenuNext,
            ]),
        );
        let edit_mode = Emacs::new(keybindings);

        let mut line_editor = Reedline::create()
            .with_history(history)
            .with_hinter(Box::new(hinter))
            .with_completer(Box::new(completer))
            .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
            .with_edit_mode(Box::new(edit_mode))
            .with_validator(Box::new(ChiliValidator {
                use_chili_syntax: !args.pepper,
            }))
            .use_bracketed_paste(true);
        // Wire the external printer only when logging to the REPL (no --dir flag).
        // When --dir is set, logs go to a file and we skip the printer.
        if let Some(p) = printer {
            line_editor = line_editor.with_external_printer(p);
        }

        let state_input = Arc::clone(&arc_state);
        let src_path = if args.pepper { "repl.pep" } else { "repl.chi" };
        let rl_handle = thread::spawn(move || {
            let state = state_input;
            loop {
                let readline = line_editor.read_line(&prompt);
                match readline {
                    Ok(Signal::Success(line)) => {
                        if line.is_empty() {
                            continue;
                        }
                        if line == "\\\\" {
                            break;
                        }
                        let start = Instant::now();
                        let nodes = match state.parse("", &line) {
                            Ok(nodes) => nodes,
                            Err(e) => {
                                eprintln!("\x1b[1;91m{}\x1b[0m", e);
                                continue;
                            }
                        };
                        let state = state.clone();
                        let handle = thread::spawn(move || {
                            match state.eval_ast(nodes.clone(), src_path, &line) {
                                Ok(any) => {
                                    println!("\x1b[1;90m{:?}\x1b[0m", start.elapsed());
                                    println!("{}", any);
                                }
                                Err(e) => eprintln!("\x1b[1;91m{}\x1b[0m", e),
                            }
                        });
                        handle.join().unwrap_or_else(|e| {
                            eprintln!("\x1b[1;91m{:?}\x1b[0m", e);
                        });
                    }
                    Ok(Signal::CtrlC) => {
                        println!("CTRL-C");
                        break;
                    }
                    Ok(Signal::CtrlD) => {
                        println!("CTRL-D");
                        break;
                    }
                    Ok(Signal::ExternalBreak(_)) => {
                        println!("External break");
                        break;
                    }
                    Ok(s) => {
                        println!("Unknown signal: {:?}", s);
                        break;
                    }
                    Err(err) => {
                        println!("Error: {:?}", err);
                        break;
                    }
                }
            }
        });

        match rl_handle.join() {
            Ok(_) => {
                info!("Exiting ...");
            }
            Err(e) => {
                error!("Exiting ... error {:?}", e);
                exit(1);
            }
        }
    }
    Ok(())
}
