mod completer;
mod logger;
mod pipe;
mod validator;

use crate::logger::LOG_FN;
use crate::pipe::Pipe;
use crate::validator::ChiliValidator;
use chili_core::{ConnType, EngineState, IpcType, utils};
use chili_op::BUILT_IN_FN;
use clap::Parser;
use crossterm::cursor::Show;
use crossterm::execute;
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};
use env_logger::Target;
use home::home_dir;
use log::{debug, error, info, warn};
use sysinfo::Pid;

use crate::completer::ChiliCompleter;
use nu_ansi_term::{Color, Style};
use reedline::{
    ColumnarMenu, DefaultHinter, DefaultPrompt, DefaultPromptSegment, Emacs, ExternalPrinter,
    FileBackedHistory, KeyCode, KeyModifiers, MenuBuilder, Reedline, ReedlineEvent, ReedlineMenu,
    Signal, default_emacs_keybindings,
};
use std::fs::File;
use std::io::{Write, stdout};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::exit;
use std::str::FromStr;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let log_level = log::LevelFilter::from_str(&args.log_level.to_lowercase())
        .unwrap_or(log::LevelFilter::Info);

    let debug = log_level == log::LevelFilter::Debug;

    let printer = ExternalPrinter::default();
    let pipe = Pipe::new(printer.clone());

    let target = if !args.log_dir.is_empty() {
        let log_dir = PathBuf::from(&args.log_dir);
        if !log_dir.exists()
            && let Err(e) = std::fs::create_dir_all(&log_dir)
        {
            println!("failed to create log directory: {}", e);
            exit(1);
        }
        let log_file = log_dir.join(format!(
            "chili_{}.log",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        ));
        match File::create(log_file) {
            Ok(file) => Target::Pipe(Box::new(file)),
            Err(e) => {
                println!("failed to create log file: {}", e);
                exit(1);
            }
        }
    } else {
        Target::Pipe(Box::new(pipe))
    };

    env_logger::Builder::new()
        .filter_level(log_level)
        .format_module_path(false)
        .format_target(false)
        .format_timestamp_millis()
        .target(target)
        .init();

    #[cfg(not(feature = "vintage"))]
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

    #[cfg(feature = "vintage")]
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

    #[cfg(not(feature = "vintage"))]
    unsafe {
        std::env::set_var("CHILI_SYNTAX", "chili")
    };

    if args.memory_limit > 0.0 {
        let memory_limit = if args.memory_limit > 1024.0 {
            args.memory_limit
        } else {
            info!(
                "Memory limit is set to {:>6.2} MB, which is less than 1024 MB, rounding up to 1024 MB",
                args.memory_limit
            );
            1024.0
        };
        unsafe { std::env::set_var("CHILI_MEMORY_LIMIT", memory_limit.to_string()) };
        thread::spawn(move || {
            check_memory_usage(
                memory_limit,
                sysinfo::get_current_pid().expect("Failed to get current pid"),
            )
        });
    }

    let state = EngineState::new(debug, args.lazy);

    if debug {
        info!("Debug mode is enabled");
    }

    if args.lazy {
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

    let completer = ChiliCompleter::new(&arc_state);

    if args.port > 0 {
        info!("listening at port {}", args.port);
        let state_tcp = Arc::clone(&arc_state);
        let addr = if args.remote {
            format!("0.0.0.0:{}", args.port)
        } else {
            format!("127.0.0.1:{}", args.port)
        };
        thread::spawn(move || {
            let listener = match TcpListener::bind(addr) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{} - {}", e, args.port);
                    exit(1)
                }
            };

            for stream in listener.incoming() {
                let state_tcp = Arc::clone(&state_tcp);
                let mut stream = stream.unwrap();
                let auth_info = state_tcp.validate_auth_token(&mut stream, &args.users);
                if !auth_info.is_authenticated {
                    info!(
                        "{}@{} failed to authenticate, disconnecting...",
                        auth_info.username,
                        stream.peer_addr().unwrap()
                    );
                    stream.shutdown(std::net::Shutdown::Both).unwrap();
                    continue;
                }
                info!(
                    "{}@{} connected",
                    auth_info.username,
                    stream.peer_addr().unwrap()
                );
                if auth_info.version <= 6 {
                    stream.write_all(&[6]).unwrap();
                } else {
                    stream.write_all(&[9]).unwrap();
                }
                // if not set, small package will be pending for 40ms
                stream.set_nodelay(true).unwrap();
                let peer_addr = stream.peer_addr().unwrap().to_string();
                let h = state_tcp
                    .set_handle(
                        Some(Box::new(stream.try_clone().unwrap())),
                        &peer_addr,
                        &format!(
                            "{}://{}",
                            IpcType::from_u8(auth_info.version).unwrap(),
                            peer_addr,
                        ),
                        false,
                        IpcType::from_u8(auth_info.version).unwrap(),
                        ConnType::Incoming,
                        0,
                    )
                    .unwrap();
                if auth_info.version <= 6 {
                    let mut stream = Box::new(stream);
                    thread::spawn(move || {
                        utils::handle_q_conn(
                            &mut stream,
                            peer_addr.starts_with("127.0.0.1"),
                            h.to_i64().unwrap(),
                            state_tcp,
                            &auth_info.username,
                        )
                    });
                } else {
                    thread::spawn(move || {
                        utils::handle_chili_conn(
                            &mut stream,
                            peer_addr.starts_with("127.0.0.1"),
                            h.to_i64().unwrap(),
                            state_tcp,
                            &auth_info.username,
                        )
                    });
                }
            }
        });
    }

    if args.src.is_some()
        && let Err(e) = arc_state.load_source_path("", &args.src.unwrap())
    {
        eprintln!("\x1b[1;91m{}\x1b[0m", e);
        if !debug {
            exit(1);
        }
    }

    let job_state = Arc::clone(&arc_state);

    if args.interval > 0 {
        thread::spawn(move || {
            loop {
                debug!("executing jobs");
                job_state.execute_jobs();
                thread::sleep(Duration::from_millis(args.interval));
            }
        });
    }

    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("* ".to_owned()),
        DefaultPromptSegment::Empty,
    );

    let history_path = home_dir().unwrap().join(".chili_history");
    let history = Box::new(
        FileBackedHistory::with_file(1000, history_path).expect("Failed to create history file"),
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
        .with_validator(Box::new(ChiliValidator {}))
        .with_external_printer(printer)
        .use_bracketed_paste(true);

    let state_input = Arc::clone(&arc_state);
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
                    let handle =
                        thread::spawn(move || match state.eval_ast(nodes.clone(), "", &line) {
                            Ok(any) => {
                                println!("\x1b[1;90m{:?}\x1b[0m", start.elapsed());
                                println!("{}", any);
                            }
                            Err(e) => eprintln!("\x1b[1;91m{}\x1b[0m", e),
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
    Ok(())
}

fn check_memory_usage(memory_limit: f64, pid: Pid) {
    let mut is_over_limit = false;
    loop {
        let sys = sysinfo::System::new_all();
        let process = sys.process(pid);
        if let Some(process) = process {
            let memory_usage = process.memory();
            let memory_usage_mb = memory_usage as f64 / 1048576.0;
            if memory_usage_mb > memory_limit {
                // only quit if already over the limit last check
                if is_over_limit {
                    eprintln!(
                        "Memory usage exceeded the limit: {:>6.2} MB, usage: {:>6.2} MB, exiting...",
                        memory_limit, memory_usage_mb
                    );
                    let mut stdout = stdout();
                    // Perform the standard cleanup sequence
                    let _ = disable_raw_mode();
                    let _ = execute!(stdout, LeaveAlternateScreen, Show);
                    let _ = stdout.flush();
                    exit(1);
                } else {
                    warn!(
                        "Memory usage exceeded the limit: {:>6.2} MB, usage: {:>6.2} MB, will exit if the next check also exceeds the limit",
                        memory_limit, memory_usage_mb
                    );
                    is_over_limit = true;
                }
            } else {
                if memory_usage_mb > memory_limit * 0.9 {
                    warn!(
                        "Memory usage is reaching the 90% limit: {:>6.2} MB, usage: {:>6.2} MB",
                        memory_limit, memory_usage_mb
                    );
                }
                is_over_limit = false;
            }
        } else {
            warn!("Process {} not found, stopping memory limit check", pid);
            break;
        }
        thread::sleep(Duration::from_secs(3));
    }
}
