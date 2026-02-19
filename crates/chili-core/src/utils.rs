use std::{
    env,
    io::{Read, Write},
    net::TcpStream,
    sync::Arc,
    time::Duration,
};

use crate::{errors::SpicyError, obj::SpicyObj};
use log::{debug, error, info};

use crate::{EngineState, Stack, engine_state::ReadWrite, serde6, serde9};

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
    (message_type, len + ((header[3] as usize) << 32), header[2])
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
        let res = state.eval(&mut stack, &obj);
        debug!("evaluated result: {:?}", res);

        if message_type == MessageType::Sync {
            match res {
                Ok(obj) => {
                    let mut v8 = serde6::serialize(&obj).unwrap();
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
        let mut res = state.eval(&mut Stack::default(), &f);
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
            res = state.eval(&mut Stack::default(), &f);
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

        debug!("eval chili IPC message: {:?}", any);
        stack.clear_vars();
        let res = state.eval(&mut stack, &any);

        if message_type == MessageType::Sync {
            match res {
                Ok(obj) => {
                    let v8 = serde9::serialize(&obj, !is_local).unwrap();
                    let _ = crate::write_chili_ipc_msg(rw, &v8, MessageType::Response);
                }
                Err(e) => {
                    let err = serde9::serialize_err(&e.to_string());
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
        let mut res = state.eval(&mut Stack::default(), &f);
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
            res = state.eval(&mut Stack::default(), &f);
            if retry < 6 {
                retry += 1;
            }
        }
    }
}
