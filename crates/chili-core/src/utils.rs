use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    sync::{Arc, LazyLock},
    time::Duration,
};

use crate::{errors::SpicyError, obj::SpicyObj};
use log::{debug, error, info, warn};
use polars::{frame::DataFrame, prelude::IntoColumn, series::Series};
use regex::Regex;

use crate::{ConnType, EngineState, Stack, engine_state::ReadWrite, serde6, serde9};

/// A thin wrapper around [`std::fs::File`] that makes [`Write::flush`] call
/// [`File::sync_data`] (`fdatasync`), so that an explicit `.flush()` guarantees
/// data durability on disk.  Normal `write()` calls pass straight through
/// without any extra syscalls.
///
/// On a sticky EIO (or any non-EINTR flush error), `flush()` closes the
/// underlying fd, reopens the file at the same path, seeks to EOF, and
/// retries `sync_data()` once.  This matches the recovery pattern used by
/// Postgres, etcd, and RocksDB for transient device-level writeback errors.
struct SyncFile {
    file: std::fs::File,
    path: PathBuf,
}

impl Write for SyncFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.file.sync_data() {
            Ok(()) => return Ok(()),
            Err(e) if e.raw_os_error() == Some(4 /* EINTR */) => {
                // EINTR — retry once in place, no reopen needed.
                return self.file.sync_data();
            }
            Err(first_err) => {
                // Persistent error (likely sticky EIO) — close and reopen the fd.
                let path_display = self.path.display();
                let new_file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(false)
                    .open(&self.path)
                    .map_err(|reopen_err| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!(
                                "fsync failed ({first_err}), reopen of {path_display} also failed: {reopen_err}"
                            ),
                        )
                    })?;
                self.file = new_file;
                // Seek to end so subsequent writes append correctly.
                use std::io::{Seek, SeekFrom};
                self.file.seek(SeekFrom::End(0))?;
                warn!(
                    "fsync EIO on {}: closed and reopened fd after transient error",
                    path_display
                );
                self.file.sync_data().map_err(|retry_err| {
                    std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!(
                            "fsync failed ({first_err}), retried after reopen of {path_display}, still failed: {retry_err}"
                        ),
                    )
                })
            }
        }
    }
}

impl Read for SyncFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

/// Detect the [`ConnType`] of an already-open file by inspecting its size
/// and magic header bytes.
///
/// - Empty file → `ConnType::New`
/// - Non-empty with `[255, 0, 0, 0]` magic prefix → `ConnType::Sequence`
/// - Anything else → `ConnType::File`
///
/// The file cursor is left at the position right after the header read
/// (byte 4 for `Sequence`/`File` with ≥ 8 bytes, unchanged for `New`).
pub fn detect_conn_type(file: &mut std::fs::File) -> Result<ConnType, SpicyError> {
    use std::io::Read;

    let metadata = file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    if metadata.len() == 0 {
        Ok(ConnType::New)
    } else if metadata.len() < 8 {
        Ok(ConnType::File)
    } else {
        let mut header = [0u8; 4];
        file.read_exact(&mut header)
            .map_err(|e| SpicyError::Err(format!("failed to read header, error: {}", e)))?;
        if [255, 0, 0, 0] == header {
            Ok(ConnType::Sequence)
        } else {
            Ok(ConnType::File)
        }
    }
}

/// Walk the frames of a sequence file and count valid messages.
///
/// The file must be positioned right after the 8-byte magic header (i.e.
/// at byte 8). Each frame is `[8-byte size][8-byte timestamp][payload]`.
///
/// When `must_deserialize` is `true`, every payload is decoded with
/// `serde9::deserialize`; a decode failure stops the walk (the frame is
/// considered torn/corrupt). When `false`, payloads are skipped via seek.
///
/// Returns `(message_count, valid_byte_size)` where `valid_byte_size`
/// includes the 8-byte magic header and all complete frames.
pub fn count_seq_messages(
    file: &mut std::fs::File,
    must_deserialize: bool,
) -> Result<(i64, u64), SpicyError> {
    use std::io::{Read, Seek};

    let total_size = file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .len();
    let mut count: i64 = 0;
    let mut valid_size: u64 = 8; // 8-byte magic header already consumed
    while valid_size < total_size {
        let mut header = [0u8; 16];
        if file.read_exact(&mut header).is_err() {
            break;
        }
        let msg_size = u64::from_le_bytes(header[..8].try_into().unwrap());
        if msg_size == 0 {
            break;
        }
        if must_deserialize {
            let mut buffer = vec![0u8; msg_size as usize];
            if file.read_exact(&mut buffer).is_err() {
                break;
            }
            if crate::serde9::deserialize(&buffer, &mut 0).is_err() {
                break;
            }
            // Defense-in-depth: if deserialize ever panics (e.g. on a torn/truncated
            // frame), catch it here so the walk stops at the last good frame instead
            // of killing the thread. The primary fix is in serde9::deserialize itself
            // (returns Err, never panics), but this guard matches the pattern in
            // replay_chili_msgs_log for robustness.
            // Note: the Err-based check above handles the normal case; this comment
            // documents the rationale for why catch_unwind is NOT needed here after
            // the serde9 bounds-checking fix. If a future regression reintroduces
            // panics, add catch_unwind back.
        } else if file.seek_relative(msg_size as i64).is_err() {
            break;
        }
        count += 1;
        valid_size += msg_size + 16;
    }
    Ok((count, valid_size))
}

/// Strict variant of [`count_seq_messages`]: returns `Err` on any corrupt or
/// truncated frame instead of silently stopping. Use this when you want to
/// surface corruption to the caller rather than tolerating it.
pub fn count_seq_messages_strict(
    file: &mut std::fs::File,
    must_deserialize: bool,
) -> Result<(i64, u64), SpicyError> {
    use std::io::{Read, Seek};

    let total_size = file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .len();
    let mut count: i64 = 0;
    let mut valid_size: u64 = 8; // 8-byte magic header already consumed
    while valid_size < total_size {
        let mut header = [0u8; 16];
        file.read_exact(&mut header).map_err(|e| {
            SpicyError::Err(format!(
                "failed to read frame header at offset {}: {}",
                valid_size, e
            ))
        })?;
        let msg_size = u64::from_le_bytes(header[..8].try_into().unwrap());
        if msg_size == 0 {
            return Err(SpicyError::Err(format!(
                "zero-length frame at offset {}",
                valid_size
            )));
        }
        if must_deserialize {
            let mut buffer = vec![0u8; msg_size as usize];
            file.read_exact(&mut buffer).map_err(|e| {
                SpicyError::Err(format!(
                    "truncated frame payload at offset {} (expected {} bytes): {}",
                    valid_size, msg_size, e
                ))
            })?;
            crate::serde9::deserialize(&buffer, &mut 0).map_err(|e| {
                SpicyError::Err(format!(
                    "corrupt frame at offset {} (message #{}): {}",
                    valid_size, count, e
                ))
            })?;
        } else {
            file.seek_relative(msg_size as i64).map_err(|e| {
                SpicyError::Err(format!(
                    "failed to seek past frame at offset {}: {}",
                    valid_size, e
                ))
            })?;
        }
        count += 1;
        valid_size += msg_size + 16;
    }
    Ok((count, valid_size))
}

/// Open a `file://` path for writing: open (read+write+create, no truncate),
/// detect `ConnType` from existing content (`New`/`File`/`Sequence`),
/// then seek to EOF so subsequent writes append.
///
/// The returned writer is a [`SyncFile`], so calling `.flush()` on it will
/// issue `fdatasync` to ensure data durability.
///
/// Returns `(writer, conn_type, msg_count)` where `msg_count` is the
/// number of valid messages in the file (only non-zero for `Sequence` files).
pub fn prepare_file_writer(path: &str) -> Result<(Box<dyn ReadWrite>, ConnType, i64), SpicyError> {
    use std::fs;
    use std::io::{Read, Seek, SeekFrom};

    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    let conn_type = detect_conn_type(&mut file)?;
    let msg_count = if conn_type == ConnType::Sequence {
        // detect_conn_type read 4 bytes; skip the remaining 4 of the 8-byte magic header
        let mut pad = [0u8; 4];
        file.read_exact(&mut pad)
            .map_err(|e| SpicyError::Err(format!("failed to read header, error: {}", e)))?;
        let (count, valid_size) = count_seq_messages(&mut file, false)?;
        // Truncate any torn/partial trailing frame so new writes start
        // right after the last valid message.
        file.set_len(valid_size)
            .map_err(|e| SpicyError::Err(format!("failed to truncate file, error: {}", e)))?;
        file.seek(SeekFrom::Start(valid_size))
            .map_err(|e| SpicyError::Err(format!("failed to seek, error: {}", e)))?;
        count
    } else {
        file.seek(SeekFrom::End(0))
            .map_err(|e| SpicyError::Err(format!("failed to seek to end of file, error: {}", e)))?;
        0
    };
    let rw: Box<dyn ReadWrite> = Box::new(SyncFile {
        file,
        path: PathBuf::from(path),
    });
    Ok((rw, conn_type, msg_count))
}

pub fn unpack_socket(socket: &str) -> Result<(String, String, String, String), SpicyError> {
    let sockets = socket.splitn(4, ":").collect::<Vec<&str>>();
    if sockets.len() < 2 {
        return Err(SpicyError::EvalErr(
            "Required at least a host and a port".to_string(),
        ));
    }
    let host = if sockets[0].is_empty() {
        "localhost"
    } else {
        sockets[0]
    };
    let port = sockets[1];

    if port.is_empty() {
        return Err(SpicyError::EvalErr(
            "Required a port, got empty port".to_string(),
        ));
    }

    let password = if sockets.len() == 4 && !sockets[3].is_empty() {
        sockets[3].to_string()
    } else {
        env::var("CHILI_IPC_TOKEN").unwrap_or_default()
    };

    let user = if sockets.len() >= 3 && !sockets[2].is_empty() {
        sockets[2].to_string()
    } else {
        whoami::username().unwrap_or_default()
    };

    Ok((host.to_string(), port.to_string(), user, password))
}

pub fn send_auth(
    stream: &mut TcpStream,
    user: &str,
    password: &str,
    version: u8,
) -> Result<u8, SpicyError> {
    let mut buf = format!("{}:{}", user, password).as_bytes().to_vec();
    // version 9 for chili ipc format
    buf.push(version);
    buf.push(0);
    stream
        .write_all(&buf)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    let mut version = [0u8; 1];
    stream
        .read_exact(&mut version)
        .map_err(|_| SpicyError::Err("authentication failed, wrong credentials".to_owned()))?;
    Ok(version[0])
}

pub fn read_q_msg(
    rw: &mut dyn ReadWrite,
    length: usize,
    compression_mode: u8,
) -> Result<SpicyObj, SpicyError> {
    let mut vec = vec![0u8; length];
    rw.read_exact(&mut vec)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    if compression_mode == 1 {
        let length = u32::from_le_bytes(vec[..4].try_into().unwrap()) as usize;
        let mut de_vec = vec![0u8; length - 8];
        serde6::decompress(&vec, &mut de_vec, 4);
        Ok(serde6::deserialize(&de_vec, &mut 0, false)?)
    } else if compression_mode == 2 {
        let length = u64::from_le_bytes(vec[..8].try_into().unwrap()) as usize;
        let mut de_vec = vec![0u8; length - 8];
        serde6::decompress(&vec, &mut de_vec, 8);
        Ok(serde6::deserialize(&de_vec, &mut 0, false)?)
    } else {
        Ok(serde6::deserialize(&vec, &mut 0, false)?)
    }
}

pub fn read_chili_ipc_msg(rw: &mut dyn ReadWrite, length: usize) -> Result<SpicyObj, SpicyError> {
    let mut vec = vec![0u8; length];
    rw.read_exact(&mut vec)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    serde9::deserialize(&vec, &mut 0)
}

pub fn read_q_table_name(msg: &[u8]) -> Result<String, SpicyError> {
    if msg[0] != 0 {
        return Err(SpicyError::Err(format!(
            "Expected mixed list message, got type {}",
            msg[0]
        )));
    }
    if msg[2..6] != [3, 0, 0, 0] {
        return Err(SpicyError::Err(format!(
            "Expected mixed list message with 3 items, got {} items",
            u32::from_le_bytes(msg[3..7].try_into().unwrap())
        )));
    }
    if msg[6] == 245 || msg[6] == 10 {
        let mut pos = 6;
        match msg[6] {
            245 => {
                pos += 1;
                while msg[pos] != 0 {
                    pos += 1;
                }
                pos += 1;
            }
            10 => {
                let size = u32::from_le_bytes(msg[8..12].try_into().unwrap());
                pos = 12 + size as usize;
            }
            _ => unreachable!(),
        }
        let table_name = serde6::deserialize(msg, &mut pos, false)?;
        Ok(table_name.str()?.to_string())
    } else {
        Err(SpicyError::Err(
            "The first item is not a (symbol|string)".to_string(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    Async = 0,
    Sync = 1,
    Response = 2,
}

impl MessageType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(MessageType::Async),
            1 => Some(MessageType::Sync),
            2 => Some(MessageType::Response),
            _ => None,
        }
    }
}

pub fn decode_header6(header: &[u8]) -> (MessageType, usize, u8) {
    let message_type = MessageType::from_u8(header[1]).unwrap();
    let len = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
    (
        message_type,
        len + ((header[3] as usize).wrapping_shl(32)),
        header[2],
    )
}

pub fn decode_header9(header: &[u8]) -> (MessageType, usize) {
    let message_type = MessageType::from_u8(header[1]).unwrap();
    let len = u64::from_le_bytes(header[8..].try_into().unwrap());
    (message_type, len as usize)
}

pub fn write_q_ipc_msg(
    rw: &mut Box<dyn ReadWrite>,
    buf: &[u8],
    message_type: MessageType,
) -> Result<(), std::io::Error> {
    // little endian 1, sync 1, 0, 0
    rw.write_all(&[1, message_type as u8, 0, 0])?;
    rw.write_all(&((buf.len() + 8) as u32).to_le_bytes())?;
    rw.write_all(buf)?;
    Ok(())
}

pub fn write_chili_ipc_msg(
    rw: &mut dyn ReadWrite,
    buf: &[Vec<u8>],
    message_type: MessageType,
) -> Result<(), std::io::Error> {
    let len = buf.iter().map(|v| v.len()).sum::<usize>();
    rw.write_all(&[1, message_type as u8, 0, 0, 0, 0, 0, 0])?;
    rw.write_all(&(len as u64).to_le_bytes())?;
    buf.iter().try_for_each(|v| rw.write_all(v))?;
    Ok(())
}

static RE_STYLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1B\[[0-9;]*m").unwrap());

pub fn handle_q_conn(
    rw: &mut dyn ReadWrite,
    is_local: bool,
    handle: i64,
    state: Arc<EngineState>,
    user: &str,
) {
    let mut header = [0u8; 8];
    let mut stack = Stack::new(None, 0, handle, user);
    loop {
        // little endian, msg type()
        if let Err(e) = rw.read_exact(&mut header) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                info!("publisher disconnected, handle {}", handle);
            } else {
                error!(
                    "failed to read from publisher with error {}, disconnecting",
                    e
                );
            }
            break;
        }
        let len = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        let message_type = MessageType::from_u8(header[1]).unwrap();
        let obj = match crate::read_q_msg(rw, len - 8, header[2]) {
            Ok(obj) => obj,
            Err(e) => {
                if message_type == MessageType::Sync
                    && let Err(_) =
                        rw.write_all(&serde6::serialize(&SpicyObj::Err(e.to_string())).unwrap())
                {
                    break;
                }
                continue;
            }
        };

        debug!("evaluate q IPC message: {:?}", obj);
        stack.clear_vars();
        let src_path = if state.is_repl_use_chili_syntax() {
            format!("ipc{}.chi", handle)
        } else {
            format!("ipc{}.pep", handle)
        };
        let res = state.eval(&mut stack, &obj, &src_path);
        debug!("evaluated result: {:?}", res);

        if message_type == MessageType::Sync {
            match res {
                Ok(obj) => match serde6::serialize(&obj) {
                    Ok(mut v8) => {
                        if !is_local {
                            v8 = serde6::compress(v8);
                        }
                        let _ = rw.write(&[1, 2, 0, 0]);
                        let _ = rw.write_all(&((v8.len() + 8) as u32).to_le_bytes());
                        let _ = rw.write_all(&v8);
                    }
                    Err(e) => {
                        let err = serde6::serialize(&SpicyObj::Err(e.to_string())).unwrap();
                        let _ = rw.write_all(&[1, 2, 0, 0]);
                        let _ = rw.write_all(&(err.len() as u32 + 8).to_le_bytes());
                        let _ = rw.write_all(&err);
                    }
                },
                Err(e) => {
                    let err_msg = RE_STYLE.replace_all(&e.to_string(), "").to_string();
                    let err = serde6::serialize(&SpicyObj::Err(err_msg)).unwrap();
                    let _ = rw.write_all(&[1, 2, 0, 0]);
                    let _ = rw.write_all(&(err.len() as u32 + 8).to_le_bytes());
                    let _ = rw.write_all(&err);
                }
            }
        } else if let Err(e) = res {
            error!("{}", e);
        }
    }

    let _ = state.disconnect_handle(&handle);
    if let Ok(callback) = state.get_callback(&handle)
        && !callback.is_empty()
    {
        info!("calling '{}' function for handle {}", &callback, handle);
        let f = SpicyObj::MixedList(vec![
            SpicyObj::Symbol(callback.clone()),
            SpicyObj::I64(handle),
        ]);
        let mut res = state.eval(&mut Stack::default(), &f, "");
        let mut retry = 1;
        while res.is_err() {
            let delay = 2_u64.pow(retry);
            error!(
                "failed to call '{}' function for handle {}, retrying in {} seconds\n{}",
                &callback,
                handle,
                delay,
                res.err().unwrap(),
            );
            std::thread::sleep(Duration::from_secs(delay));
            res = state.eval(&mut Stack::default(), &f, "");
            if retry < 6 {
                retry += 1;
            }
        }
    }
}

pub fn handle_chili_conn(
    rw: &mut dyn ReadWrite,
    is_local: bool,
    handle: i64,
    state: Arc<EngineState>,
    user: &str,
) {
    let mut header = [0u8; 16];
    let mut stack = Stack::new(None, 0, handle, user);
    loop {
        // little endian, msg type()
        if let Err(e) = rw.read_exact(&mut header) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                info!("publisher disconnected, handle {}", handle);
            } else {
                error!(
                    "failed to read from publisher with error {}, disconnecting",
                    e
                );
            }
            break;
        }
        let (message_type, len) = crate::utils::decode_header9(&header);
        let any = match crate::read_chili_ipc_msg(rw, len) {
            Ok(obj) => obj,
            Err(e) => {
                if message_type == MessageType::Sync
                    && rw
                        .write_all(&serde9::serialize_err(&e.to_string()))
                        .is_err()
                {
                    break;
                }
                continue;
            }
        };

        let src_path = if state.is_repl_use_chili_syntax() {
            format!("ipc{}.chi", handle)
        } else {
            format!("ipc{}.pep", handle)
        };
        debug!("eval chili IPC message: {:?}", any);
        stack.clear_vars();
        let res = state.eval(&mut stack, &any, &src_path);

        if message_type == MessageType::Sync {
            match res {
                Ok(obj) => match serde9::serialize(&obj, !is_local) {
                    Ok(v8) => {
                        let _ = crate::write_chili_ipc_msg(rw, &v8, MessageType::Response);
                    }
                    Err(e) => {
                        let err = serde9::serialize_err(&e.to_string());
                        let _ = rw.write_all(&err);
                        error!("failed to serialize response: {}", e);
                    }
                },
                Err(e) => {
                    let err_msg = RE_STYLE.replace_all(&e.to_string(), "").to_string();
                    let err = serde9::serialize_err(&err_msg);
                    let _ = rw.write_all(&err);
                }
            }
        } else if let Err(e) = res {
            error!("{}", e);
        }
    }

    let _ = state.disconnect_handle(&handle);
    if let Ok(callback) = state.get_callback(&handle)
        && !callback.is_empty()
    {
        info!("calling '{}' function for handle {}", &callback, handle);
        let f = SpicyObj::MixedList(vec![
            SpicyObj::Symbol(callback.clone()),
            SpicyObj::I64(handle),
        ]);
        let mut res = state.eval(&mut Stack::default(), &f, "");
        let mut retry = 1;
        while res.is_err() {
            let delay = 2_u64.pow(retry);
            error!(
                "failed to call '{}' function for handle {}, retrying in {} seconds\n{}",
                &callback,
                handle,
                delay,
                res.err().unwrap(),
            );
            std::thread::sleep(Duration::from_secs(delay));
            res = state.eval(&mut Stack::default(), &f, "");
            if retry < 6 {
                retry += 1;
            }
        }
    }
}

pub fn convert_list_to_df(list: &[SpicyObj], df: &DataFrame) -> Result<DataFrame, SpicyError> {
    let series = list
        .iter()
        .map(|args| args.as_series())
        .collect::<Result<Vec<Series>, SpicyError>>()?;
    if series.len() > df.width() {
        return Err(SpicyError::Err(
            "number of columns in list is greater than number of columns in dataframe".to_string(),
        ));
    }
    let height = series.first().map(|s| s.len()).unwrap_or(0);
    let column_names = df.get_column_names_owned();
    DataFrame::new(
        height,
        series
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                s.clone()
                    .rename(column_names[i].clone())
                    .clone()
                    .into_column()
            })
            .collect::<Vec<_>>(),
    )
    .map_err(|e| SpicyError::Err(e.to_string()))
}
