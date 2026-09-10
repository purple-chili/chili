use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{Arc, LazyLock, mpsc},
    time::Duration,
};

use crate::{errors::SpicyError, errors::SpicyResult, obj::SpicyObj};
use log::{debug, error, info, warn};
use polars::{
    frame::DataFrame,
    prelude::{DataType, IntoColumn, IntoSeries, TimeZone},
    series::Series,
};
use regex::Regex;

use crate::{ConnType, EngineState, Stack, engine_state::ReadWrite, serde6, serde9};

const SEQ_MAGIC: [u8; 8] = [255, 0, 0, 0, 0, 0, 0, 0];
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

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

/// True when the file starts with gzip magic (`1f 8b`). Rewinds to offset 0.
pub fn file_starts_with_gzip(file: &mut std::fs::File) -> Result<bool, SpicyError> {
    use std::io::{Read, Seek, SeekFrom};

    file.seek(SeekFrom::Start(0))
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    let mut magic = [0u8; 2];
    let is_gzip = match file.read_exact(&mut magic) {
        Ok(()) => magic == GZIP_MAGIC,
        Err(_) => false,
    };
    file.seek(SeekFrom::Start(0))
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(is_gzip)
}

/// Detect the [`ConnType`] of an already-open file by inspecting its size
/// and magic header bytes. Supports whole-file gzip sequence archives.
///
/// - Empty file → `ConnType::New`
/// - Non-empty with `[255, 0, 0, 0]` magic prefix (plain or inside gzip) → `ConnType::Sequence`
/// - Anything else → `ConnType::File`
///
/// The file cursor is left at the position right after the header read
/// (byte 4 for plain `Sequence`/`File` with ≥ 8 bytes, unchanged for `New` or gzip).
pub fn detect_conn_type(file: &mut std::fs::File) -> Result<ConnType, SpicyError> {
    use std::io::Read;

    let metadata = file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    if metadata.len() == 0 {
        return Ok(ConnType::New);
    }
    if file_starts_with_gzip(file)? {
        use flate2::read::GzDecoder;
        let probe = file
            .try_clone()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let mut decoder = GzDecoder::new(probe);
        let mut header = [0u8; 8];
        return match decoder.read_exact(&mut header) {
            Ok(()) if header == SEQ_MAGIC => Ok(ConnType::Sequence),
            Ok(()) => Ok(ConnType::File),
            Err(_) => Ok(ConnType::File),
        };
    }
    if metadata.len() < 8 {
        return Ok(ConnType::File);
    }
    let mut header = [0u8; 4];
    file.read_exact(&mut header)
        .map_err(|e| SpicyError::Err(format!("failed to read header, error: {}", e)))?;
    if [255, 0, 0, 0] == header {
        Ok(ConnType::Sequence)
    } else {
        Ok(ConnType::File)
    }
}

/// Walk sequence frames on any reader positioned after the 8-byte magic header.
/// `total_size` is the on-disk byte length for plain files (`None` for gzip streams).
fn count_seq_messages_on_reader(
    reader: &mut dyn Read,
    must_deserialize: bool,
    strict: bool,
    total_size: Option<u64>,
) -> Result<(i64, u64), SpicyError> {
    use std::io::ErrorKind;

    let mut count: i64 = 0;
    let mut valid_size: u64 = 8;
    loop {
        if let Some(total) = total_size
            && valid_size >= total
        {
            break;
        }
        let mut header = [0u8; 16];
        match reader.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof && total_size.is_none() => break,
            Err(_) if !strict => break,
            Err(e) => {
                return Err(SpicyError::Err(format!(
                    "failed to read frame header at offset {}: {}",
                    valid_size, e
                )));
            }
        }
        let msg_size = u64::from_le_bytes(header[..8].try_into().unwrap());
        if msg_size == 0 {
            if strict {
                return Err(SpicyError::Err(format!(
                    "zero-length frame at offset {}",
                    valid_size
                )));
            }
            break;
        }
        if must_deserialize {
            let mut buffer = vec![0u8; msg_size as usize];
            if let Err(e) = reader.read_exact(&mut buffer) {
                if strict {
                    return Err(SpicyError::Err(format!(
                        "truncated frame payload at offset {} (expected {} bytes): {}",
                        valid_size, msg_size, e
                    )));
                }
                break;
            }
            if let Err(e) = crate::serde9::deserialize(&buffer, &mut 0) {
                if strict {
                    return Err(SpicyError::Err(format!(
                        "corrupt frame at offset {} (message #{}): {}",
                        valid_size, count, e
                    )));
                }
                break;
            }
        } else {
            let mut discard = reader.take(msg_size);
            match std::io::copy(&mut discard, &mut std::io::sink()) {
                Ok(n) if n == msg_size => {}
                Ok(n) if strict => {
                    return Err(SpicyError::Err(format!(
                        "truncated frame payload at offset {} (expected {} bytes, got {})",
                        valid_size, msg_size, n
                    )));
                }
                Ok(_) => break,
                Err(e) if strict => {
                    return Err(SpicyError::Err(format!(
                        "failed to read past frame at offset {}: {}",
                        valid_size, e
                    )));
                }
                Err(_) => break,
            }
        }
        count += 1;
        valid_size += msg_size + 16;
    }
    Ok((count, valid_size))
}

/// Open a decompressed sequence reader after the 8-byte magic header.
fn open_sequence_payload_reader(
    file: std::fs::File,
    gzip: bool,
) -> Result<Box<dyn Read>, SpicyError> {
    use std::io::Read;

    let mut reader: Box<dyn Read> = if gzip {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| SpicyError::Err(format!("failed to read sequence header: {}", e)))?;
    if magic != SEQ_MAGIC {
        return Err(SpicyError::Err("not a sequence file".to_string()));
    }
    Ok(reader)
}

/// Count valid messages in a plain or gzip whole-file sequence archive.
pub fn count_sequence_file_messages(
    path: &str,
    must_deserialize: bool,
    strict: bool,
) -> Result<(i64, u64, bool), SpicyError> {
    use std::fs;
    use std::io::{Seek, SeekFrom};

    let mut file = fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| SpicyError::Err(format!("failed to open file '{}': {}", path, e)))?;
    if file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .len()
        == 0
    {
        return Ok((0, 0, false));
    }
    let gzip = file_starts_with_gzip(&mut file)?;
    let count_file = file
        .try_clone()
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    let mut reader = open_sequence_payload_reader(count_file, gzip)?;
    let total_size = if gzip {
        None
    } else {
        Some(
            file.metadata()
                .map_err(|e| SpicyError::Err(e.to_string()))?
                .len(),
        )
    };
    let (count, valid_size) =
        count_seq_messages_on_reader(&mut reader, must_deserialize, strict, total_size)?;
    if !gzip {
        file.seek(SeekFrom::Start(0))
            .map_err(|e| SpicyError::Err(e.to_string()))?;
    }
    Ok((count, valid_size, gzip))
}

/// Walk the frames of a sequence file and count valid messages.
///
/// The file must be positioned right after the 8-byte magic header (i.e.
/// at byte 8). Each frame is `[8-byte size][8-byte timestamp][payload]`.
///
/// When `must_deserialize` is `true`, every payload is decoded with
/// `serde9::deserialize`; a decode failure stops the walk (the frame is
/// considered torn/corrupt). When `false`, payloads are skipped via read.
///
/// Returns `(message_count, valid_byte_size)` where `valid_byte_size`
/// includes the 8-byte magic header and all complete frames.
pub fn count_seq_messages(
    file: &mut std::fs::File,
    must_deserialize: bool,
) -> Result<(i64, u64), SpicyError> {
    let total_size = file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .len();
    count_seq_messages_on_reader(file, must_deserialize, false, Some(total_size))
}

/// Strict variant of [`count_seq_messages`]: returns `Err` on any corrupt or
/// truncated frame instead of silently stopping. Use this when you want to
/// surface corruption to the caller rather than tolerating it.
pub fn count_seq_messages_strict(
    file: &mut std::fs::File,
    must_deserialize: bool,
) -> Result<(i64, u64), SpicyError> {
    let total_size = file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .len();
    count_seq_messages_on_reader(file, must_deserialize, true, Some(total_size))
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
    if file
        .metadata()
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .len()
        == 0
    {
        let rw: Box<dyn ReadWrite> = Box::new(SyncFile {
            file,
            path: PathBuf::from(path),
        });
        return Ok((rw, ConnType::New, 0));
    }

    if file_starts_with_gzip(&mut file)? {
        let (count, _, _) = count_sequence_file_messages(path, false, false)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let rw: Box<dyn ReadWrite> = Box::new(SyncFile {
            file,
            path: PathBuf::from(path),
        });
        return Ok((rw, ConnType::Sequence, count));
    }

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

/// Relabel tz tag on a Datetime series without shifting physical timestamps.
fn relabel_datetime_tz(s: &Series, target_tz: Option<TimeZone>) -> Series {
    let name = s.name().clone();
    let DataType::Datetime(tu, _) = s.dtype() else {
        return s.clone();
    };
    let tu = *tu;
    let phys = s.to_physical_repr().into_owned();
    let i64ca = phys.i64().expect("datetime physical repr is i64");
    i64ca
        .clone()
        .into_datetime(tu, target_tz)
        .into_series()
        .with_name(name)
}

/// Ensure every column has the same length (and that matches `df.height()`).
/// Used by `drain` so a torn frame is never cleared before the caller can see it.
pub fn ensure_df_rectangular(df: &DataFrame) -> SpicyResult<()> {
    let cols = df.columns();
    if cols.is_empty() {
        return Ok(());
    }
    let expected = cols[0].len();
    let mut mismatched = Vec::new();
    for c in cols {
        if c.len() != expected {
            mismatched.push(format!("{}:{}", c.name(), c.len()));
        }
    }
    if !mismatched.is_empty() || df.height() != expected {
        let heights: Vec<String> = cols
            .iter()
            .map(|c| format!("{}:{}", c.name(), c.len()))
            .collect();
        return Err(SpicyError::Err(format!(
            "dataframe is not rectangular (height={}, columns=[{}]); refusing to drain",
            df.height(),
            heights.join(", ")
        )));
    }
    Ok(())
}

/// `DataFrame::extend` mutates columns left-to-right; a mid-column dtype error
/// leaves earlier columns grown and later ones short. We extend **in place**
/// (no clone on the success path). On failure, truncate columns back to the
/// pre-extend height via `head` so the target stays rectangular.
pub fn extend_df_atomic(df: &mut DataFrame, records: &DataFrame) -> SpicyResult<()> {
    let height_before = df.height();
    match df.extend(records) {
        Ok(()) => Ok(()),
        Err(e) => {
            if df.columns().iter().any(|c| c.len() != height_before) {
                // height cache is still `height_before` on failed extend; head
                // truncates each column to that length.
                *df = df.head(Some(height_before));
            }
            Err(SpicyError::Err(format!(
                "extend failed (target left unchanged): {e}"
            )))
        }
    }
}

/// Cast incoming columns to match `target` dtypes (by name) before extend.
/// Avoids the common String→Categorical tear on symbol/fn-style columns.
pub fn coerce_extend_dtypes(target: &DataFrame, incoming: &DataFrame) -> SpicyResult<DataFrame> {
    let needs_cast = incoming.columns().iter().any(|c| {
        target
            .column(c.name())
            .map(|t| t.dtype() != c.dtype())
            .unwrap_or(false)
    });
    if !needs_cast {
        return Ok(incoming.clone());
    }

    let height = incoming.height();
    let cols = incoming
        .columns()
        .iter()
        .map(|c| {
            let Ok(t) = target.column(c.name()) else {
                return Ok(c.clone());
            };
            if c.dtype() == t.dtype() {
                return Ok(c.clone());
            }
            c.cast(t.dtype())
                .map_err(|e| {
                    SpicyError::Err(format!(
                        "cannot cast column '{}' from {} to {}: {}",
                        c.name(),
                        c.dtype(),
                        t.dtype(),
                        e
                    ))
                })
                .map(|s| s.into_column())
        })
        .collect::<SpicyResult<Vec<_>>>()?;
    DataFrame::new(height, cols).map_err(|e| SpicyError::Err(e.to_string()))
}

/// Relabel incoming datetime timezone tags to match `target` for `df.extend`.
///
/// Same time unit, different tz tag: relabel incoming columns to the target tag
/// without shifting physical timestamps. No-op when tags already match.
pub fn coerce_extend_tz(target: &DataFrame, incoming: &DataFrame) -> DataFrame {
    let needs_fix = incoming.columns().iter().any(|c| {
        let Ok(t) = target.column(c.name()) else {
            return false;
        };
        match (t.dtype(), c.dtype()) {
            (DataType::Datetime(tu_t, tz_t), DataType::Datetime(tu_i, tz_i)) => {
                tu_t == tu_i && tz_t != tz_i
            }
            _ => false,
        }
    });
    if !needs_fix {
        return incoming.clone();
    }

    let new_cols = incoming
        .columns()
        .iter()
        .map(|c| {
            let Ok(t) = target.column(c.name()) else {
                return c.clone();
            };
            match (t.dtype(), c.dtype()) {
                (DataType::Datetime(tu_t, tz_t), DataType::Datetime(tu_i, tz_i))
                    if tu_t == tu_i && tz_t != tz_i =>
                {
                    let s = c.as_materialized_series();
                    relabel_datetime_tz(s, tz_t.clone()).into_column()
                }
                _ => c.clone(),
            }
        })
        .collect::<Vec<_>>();
    DataFrame::new(incoming.height(), new_cols).expect("relabel preserves shape")
}

static RE_STYLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1B\[[0-9;]*m").unwrap());

/// Result of an inbound IPC eval, including a wall-clock timeout (`0` = off).
#[derive(Debug)]
pub enum IpcEvalResult {
    Finished(SpicyResult<SpicyObj>),
    TimedOut,
}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_owned()
    }
}

/// Run inbound IPC eval, catching Rust panics so Sync still gets an error frame.
fn eval_ipc_catch_panic(
    state: &EngineState,
    stack: &mut Stack,
    query: &SpicyObj,
    src: &str,
) -> SpicyResult<SpicyObj> {
    match catch_unwind(AssertUnwindSafe(|| state.eval_with_pre_hook(stack, query, src))) {
        Ok(result) => result,
        Err(payload) => {
            let msg = panic_payload_message(payload);
            error!("IPC eval panicked: {}", msg);
            Err(SpicyError::EvalErr(format!("eval panicked: {msg}")))
        }
    }
}

/// Run `eval_with_pre_hook` on a worker thread when `eval_timeout_ms` is set.
///
/// Panics during eval are caught and returned as `Finished(Err(...))` so a Sync
/// caller still receives a response (with or without a timeout).
pub fn eval_ipc_with_timeout(
    state: &Arc<EngineState>,
    user: &str,
    handle: i64,
    query: &SpicyObj,
    src: &str,
) -> IpcEvalResult {
    let timeout_ms = state.eval_timeout_ms();
    if timeout_ms <= 0 {
        let mut stack = Stack::new(None, 0, handle, user);
        return IpcEvalResult::Finished(eval_ipc_catch_panic(state, &mut stack, query, src));
    }

    let (tx, rx) = mpsc::sync_channel(1);
    let state = Arc::clone(state);
    let query = query.clone();
    let src = src.to_owned();
    let user = user.to_owned();
    std::thread::spawn(move || {
        let mut stack = Stack::new(None, 0, handle, &user);
        let _ = tx.send(eval_ipc_catch_panic(&state, &mut stack, &query, &src));
    });

    match rx.recv_timeout(Duration::from_millis(timeout_ms as u64)) {
        Ok(result) => IpcEvalResult::Finished(result),
        Err(mpsc::RecvTimeoutError::Timeout) => IpcEvalResult::TimedOut,
        Err(mpsc::RecvTimeoutError::Disconnected) => IpcEvalResult::Finished(Err(
            SpicyError::EvalErr("eval worker exited before returning".into()),
        )),
    }
}

pub fn handle_q_conn(
    rw: &mut dyn ReadWrite,
    is_local: bool,
    handle: i64,
    state: Arc<EngineState>,
    user: &str,
) {
    state.fire_on_conn_open_hook(user, handle);
    let mut header = [0u8; 8];
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
                let msg = e.to_string();
                state.fire_on_bad_msg_hook(handle, &msg, None);
                if message_type == MessageType::Sync
                    && let Err(_) =
                        rw.write_all(&serde6::serialize(&SpicyObj::Err(msg)).unwrap())
                {
                    break;
                }
                continue;
            }
        };

        debug!("evaluate q IPC message: {:?}", obj);
        let src_path = if state.is_repl_use_chili_syntax() {
            format!("ipc{}.chi", handle)
        } else {
            format!("ipc{}.pep", handle)
        };
        let res = match eval_ipc_with_timeout(&state, user, handle, &obj, &src_path) {
            IpcEvalResult::Finished(res) => res,
            IpcEvalResult::TimedOut => {
                let err = SpicyError::EvalErr(format!(
                    "eval timed out after {}ms",
                    state.eval_timeout_ms()
                ));
                if message_type == MessageType::Sync {
                    let err_msg = RE_STYLE.replace_all(&err.to_string(), "").to_string();
                    let err = serde6::serialize(&SpicyObj::Err(err_msg)).unwrap();
                    let _ = rw.write_all(&[1, 2, 0, 0]);
                    let _ = rw.write_all(&(err.len() as u32 + 8).to_le_bytes());
                    let _ = rw.write_all(&err);
                    state.drop_pending_subscribers(handle);
                } else {
                    error!("{}", err);
                }
                break;
            }
        };
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
                        // Subscribe handshake: go live only after Response is on the wire.
                        state.activate_subscribers(handle);
                    }
                    Err(e) => {
                        let err = serde6::serialize(&SpicyObj::Err(e.to_string())).unwrap();
                        let _ = rw.write_all(&[1, 2, 0, 0]);
                        let _ = rw.write_all(&(err.len() as u32 + 8).to_le_bytes());
                        let _ = rw.write_all(&err);
                        state.drop_pending_subscribers(handle);
                    }
                },
                Err(e) => {
                    let err_msg = RE_STYLE.replace_all(&e.to_string(), "").to_string();
                    state.fire_on_bad_msg_hook(handle, &err_msg, None);
                    let err = serde6::serialize(&SpicyObj::Err(err_msg)).unwrap();
                    let _ = rw.write_all(&[1, 2, 0, 0]);
                    let _ = rw.write_all(&(err.len() as u32 + 8).to_le_bytes());
                    let _ = rw.write_all(&err);
                    state.drop_pending_subscribers(handle);
                }
            }
        } else if let Err(e) = res {
            let msg = e.to_string();
            state.fire_on_bad_msg_hook(handle, &msg, None);
            error!("{}", e);
        }
    }

    finish_ipc_conn(&state, user, handle);
}

fn finish_ipc_conn(state: &EngineState, user: &str, handle: i64) {
    state.fire_on_conn_close_hook(user, handle);
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
    state.fire_on_conn_open_hook(user, handle);
    let mut header = [0u8; 16];
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
                let msg = e.to_string();
                state.fire_on_bad_msg_hook(handle, &msg, None);
                if message_type == MessageType::Sync
                    && rw.write_all(&serde9::serialize_err(&msg)).is_err()
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
        let res = match eval_ipc_with_timeout(&state, user, handle, &any, &src_path) {
            IpcEvalResult::Finished(res) => res,
            IpcEvalResult::TimedOut => {
                let err = SpicyError::EvalErr(format!(
                    "eval timed out after {}ms",
                    state.eval_timeout_ms()
                ));
                if message_type == MessageType::Sync {
                    let err_msg = RE_STYLE.replace_all(&err.to_string(), "").to_string();
                    let err = serde9::serialize_err(&err_msg);
                    let _ = rw.write_all(&err);
                    state.drop_pending_subscribers(handle);
                } else {
                    error!("{}", err);
                }
                break;
            }
        };

        if message_type == MessageType::Sync {
            match res {
                Ok(obj) => match serde9::serialize(&obj, !is_local) {
                    Ok(v8) => {
                        let _ = crate::write_chili_ipc_msg(rw, &v8, MessageType::Response);
                        // Subscribe handshake: go live only after Response is on the wire.
                        state.activate_subscribers(handle);
                    }
                    Err(e) => {
                        let err = serde9::serialize_err(&e.to_string());
                        let _ = rw.write_all(&err);
                        error!("failed to serialize response: {}", e);
                        state.drop_pending_subscribers(handle);
                    }
                },
                Err(e) => {
                    let err_msg = RE_STYLE.replace_all(&e.to_string(), "").to_string();
                    state.fire_on_bad_msg_hook(handle, &err_msg, None);
                    let err = serde9::serialize_err(&err_msg);
                    let _ = rw.write_all(&err);
                    state.drop_pending_subscribers(handle);
                }
            }
        } else if let Err(e) = res {
            let msg = e.to_string();
            state.fire_on_bad_msg_hook(handle, &msg, None);
            error!("{}", e);
        }
    }

    finish_ipc_conn(&state, user, handle);
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
    let target_cols = df.columns();
    let columns = series
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            let target = &target_cols[i];
            let mut s = if s.dtype() == target.dtype() {
                s
            } else {
                s.cast(target.dtype()).map_err(|e| {
                    SpicyError::Err(format!(
                        "cannot cast column '{}' from {} to {}: {}",
                        target.name(),
                        s.dtype(),
                        target.dtype(),
                        e
                    ))
                })?
            };
            Ok(s.rename(target.name().clone()).clone().into_column())
        })
        .collect::<Result<Vec<_>, SpicyError>>()?;
    DataFrame::new(height, columns).map_err(|e| SpicyError::Err(e.to_string()))
}

#[cfg(test)]
mod tz_coerce_tests {
    use super::coerce_extend_tz;
    use polars::prelude::*;

    fn dt_col(name: &str, vals: &[i64], tz: Option<&str>) -> Column {
        let ca = Int64Chunked::from_slice(name.into(), vals);
        let tz = tz.map(|t| unsafe { TimeZone::new_unchecked(t) });
        ca.into_datetime(TimeUnit::Nanoseconds, tz)
            .into_series()
            .into_column()
    }

    #[test]
    fn utc_incoming_relabelled_to_naive_target_and_extends() {
        let height = 2usize;
        let target = DataFrame::new(
            height,
            vec![
                Column::new("seq".into(), &[1i64, 2]),
                dt_col("ingest_ts", &[1_000, 2_000], None),
            ],
        )
        .unwrap();
        let incoming = DataFrame::new(
            height,
            vec![
                Column::new("seq".into(), &[3i64, 4]),
                dt_col("ingest_ts", &[3_000, 4_000], Some("UTC")),
            ],
        )
        .unwrap();

        let coerced = coerce_extend_tz(&target, &incoming);
        assert_eq!(
            coerced.column("ingest_ts").unwrap().dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, None)
        );

        let mut t = target.clone();
        t.extend(&coerced).expect("extend after tz relabel");
        assert_eq!(t.height(), 4);
        let phys = t
            .column("ingest_ts")
            .unwrap()
            .as_materialized_series()
            .to_physical_repr()
            .into_owned();
        let got: Vec<i64> = phys.i64().unwrap().into_no_null_iter().collect();
        assert_eq!(got, vec![1_000, 2_000, 3_000, 4_000]);
    }

    #[test]
    fn naive_incoming_relabelled_to_utc_target_both_directions() {
        let height = 1usize;
        let target = DataFrame::new(
            height,
            vec![dt_col("ts", &[10_000], Some("UTC"))],
        )
        .unwrap();
        let incoming = DataFrame::new(height, vec![dt_col("ts", &[20_000], None)]).unwrap();
        let coerced = coerce_extend_tz(&target, &incoming);
        assert_eq!(
            coerced.column("ts").unwrap().dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, Some(unsafe { TimeZone::new_unchecked("UTC") }))
        );
        let mut t = target.clone();
        t.extend(&coerced).expect("extend naive->UTC target");
        let phys = t
            .column("ts")
            .unwrap()
            .as_materialized_series()
            .to_physical_repr()
            .into_owned();
        let got: Vec<i64> = phys.i64().unwrap().into_no_null_iter().collect();
        assert_eq!(got, vec![10_000, 20_000]);
    }

    #[test]
    fn matching_tz_is_zero_touch_noop() {
        let height = 1usize;
        let target = DataFrame::new(height, vec![dt_col("ts", &[1i64], Some("UTC"))]).unwrap();
        let incoming = DataFrame::new(height, vec![dt_col("ts", &[2i64], Some("UTC"))]).unwrap();
        let coerced = coerce_extend_tz(&target, &incoming);
        assert_eq!(coerced.column("ts").unwrap().dtype(), incoming.column("ts").unwrap().dtype());
    }

    #[test]
    fn different_time_unit_is_left_untouched() {
        let height = 1usize;
        let target = DataFrame::new(height, vec![dt_col("ts", &[1i64], None)]).unwrap();
        let ms = Int64Chunked::from_slice("ts".into(), &[2i64])
            .into_datetime(TimeUnit::Milliseconds, Some(unsafe { TimeZone::new_unchecked("UTC") }))
            .into_series()
            .into_column();
        let incoming = DataFrame::new(height, vec![ms]).unwrap();
        let coerced = coerce_extend_tz(&target, &incoming);
        assert_eq!(
            coerced.column("ts").unwrap().dtype(),
            &DataType::Datetime(TimeUnit::Milliseconds, Some(unsafe { TimeZone::new_unchecked("UTC") }))
        );
    }
}
