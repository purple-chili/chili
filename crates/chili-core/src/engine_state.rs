use std::{
    collections::HashMap,
    env,
    fmt::Display,
    fs::{self, DirEntry, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    net::{TcpListener, TcpStream},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, LazyLock, Mutex, RwLock},
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use lru::LruCache;

use chili_parser::Language;
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use polars::{
    frame::DataFrame,
    io::SerReader,
    io::parquet::read::ParquetReader,
    prelude::{
        ArrowDataType, ArrowField, Column, DataType, IntoLazy, NamedFrom, NamedFromOwned, TimeUnit,
        col,
    },
    series::Series,
};
use polars_arrow::{
    array::{Int64Array, ListArray},
    offset::OffsetsBuffer,
};
use rayon::prelude::*;

use crate::{
    Stack,
    authinfo::AuthInfo,
    broker::BROKER_FN,
    eval::{eval_by_node, eval_fn_call, eval_op},
    io::IO_FN,
    job::{self, Job},
    obj::SpicyObj,
    par_df::{DFType, PartitionedDataFrame},
    parse, read_chili_ipc_msg, serde6, serde9,
    side_effect_fn::SIDE_EFFECT_FN,
    utils::{
        self, MessageType, convert_list_to_df, handle_chili_conn, handle_q_conn, read_q_msg,
        read_q_table_name, send_auth, unpack_socket,
    },
};

use crate::{
    ast_node::AstNode,
    errors::{SpicyError, SpicyResult},
    func::Func,
};

pub trait ReadWrite: Read + Write + Send + Sync {}

impl<T: Read + Write + Send + Sync> ReadWrite for T {}

pub struct Handle {
    pub rw: Option<Box<dyn ReadWrite>>,
    pub socket: String,
    pub uri: String,
    pub is_local: bool,
    pub ipc_type: IpcType,
    pub conn_type: ConnType,
    pub on_disconnected: Option<String>,
}

/// LRU cache size for parsed AST trees. 256 entries × ~1 KB per AST is
/// well under 1 MB total memory budget. mdata's gateway sends a small
/// number of distinct query shapes per (path, source) pair, so this is
/// more than enough headroom for the steady-state hit rate to converge
/// near 100%.
const PARSE_CACHE_CAPACITY: usize = 256;

pub struct EngineState {
    debug: bool,
    vars: RwLock<HashMap<String, SpicyObj>>,
    par_df: RwLock<HashMap<String, PartitionedDataFrame>>,
    source: RwLock<Vec<(String, String)>>,
    // handle number, rw, is_local, version, ipc type
    handle: RwLock<IndexMap<i64, Handle>>,
    tick_count: RwLock<i64>,
    job: RwLock<IndexMap<i64, Job>>,
    topic_map: RwLock<HashMap<String, Vec<i64>>>,
    arc_self: RwLock<Option<Arc<Self>>>,
    /// Proposal D — LRU parse cache. Keyed on (path, source) so the same
    /// source under different IPC handles or REPL/IPC contexts produces
    /// distinct entries (the cached AST embeds source positions referencing
    /// the original source_id, which is preserved by NOT calling set_source
    /// on a cache hit).
    parse_cache: Mutex<LruCache<(String, String), Arc<Vec<AstNode>>>>,
    user: String,
    lazy_mode: bool,
    repl_lang: Language,
}

impl Default for EngineState {
    fn default() -> Self {
        Self::initialize()
    }
}

trait FormatFn {
    fn format_call(&self, fn_name: &str, params: &[String]) -> String;
}

impl FormatFn for Language {
    fn format_call(&self, fn_name: &str, params: &[String]) -> String {
        match self {
            Language::Chili => format!("{}({})", fn_name, params.join(", ")),
            Language::Pepper => format!("{}[{}]", fn_name, params.join("; ")),
        }
    }
}

impl EngineState {
    pub fn initialize() -> Self {
        unsafe {
            env::set_var("POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW", "1");
            env::set_var("POLARS_FMT_MAX_ROWS", "50");
            env::set_var("POLARS_FMT_MAX_COLS", "16");
        };
        let source = vec![("".to_owned(), "".to_owned())];
        let mut vars = HashMap::new();
        for (k, v) in SIDE_EFFECT_FN.iter() {
            vars.insert(k.to_owned(), SpicyObj::Fn(v.clone()));
        }
        for (k, v) in IO_FN.iter() {
            vars.insert(k.to_owned(), SpicyObj::Fn(v.clone()));
        }
        for (k, v) in BROKER_FN.iter() {
            vars.insert(k.to_owned(), SpicyObj::Fn(v.clone()));
        }
        Self {
            vars: RwLock::new(vars),
            par_df: RwLock::new(HashMap::new()),
            source: RwLock::new(source),
            handle: RwLock::new(IndexMap::new()),
            tick_count: RwLock::new(0),
            job: RwLock::new(IndexMap::new()),
            topic_map: RwLock::new(HashMap::new()),
            arc_self: RwLock::new(None),
            parse_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(PARSE_CACHE_CAPACITY).unwrap(),
            )),
            debug: false,
            user: whoami::username().unwrap_or_default(),
            lazy_mode: false,
            repl_lang: Language::Chili,
        }
    }

    pub fn enable_pepper(&mut self) {
        self.repl_lang = Language::Pepper;
    }

    pub fn is_repl_use_chili_syntax(&self) -> bool {
        self.repl_lang == Language::Chili
    }

    pub fn is_lazy_mode(&self) -> bool {
        self.lazy_mode
    }

    pub fn new(debug: bool, lazy: bool, enable_pepper: bool) -> Self {
        let mut state = Self::initialize();
        state.debug = debug;
        state.lazy_mode = lazy;
        if enable_pepper {
            state.enable_pepper();
        }
        state
    }

    pub fn register_fn(&self, map: &LazyLock<HashMap<String, Func>>) {
        let mut vars = self.vars.write().unwrap();
        map.iter().for_each(|(k, v)| {
            vars.insert(k.to_owned(), SpicyObj::Fn(v.clone()));
        });
    }

    pub fn set_arc_self(&self, arc: Arc<Self>) -> SpicyResult<()> {
        let mut arc_self = self
            .arc_self
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        *arc_self = Some(arc);
        Ok(())
    }

    pub fn shutdown(&self) {
        self.handle.write().unwrap().clear();
    }

    pub fn get_displayed_vars(&self) -> SpicyResult<HashMap<String, String>> {
        let mut vars = HashMap::new();
        for (key, obj) in self.vars.read().unwrap().iter() {
            if obj.is_fn() {
                let func = obj.fn_().unwrap();
                if func.arg_num == func.params.len() {
                    vars.insert(
                        key.to_string(),
                        self.repl_lang.format_call(key, &func.params),
                    );
                }
            } else {
                vars.insert(key.to_string(), obj.to_short_string());
            }
        }
        Ok(vars)
    }

    pub fn get_var(&self, id: &str) -> Result<SpicyObj, SpicyError> {
        let vars = self
            .vars
            .read()
            .map_err(|_| SpicyError::ReadLockErr(id.to_owned()))?;
        match vars.get(id) {
            Some(obj) => Ok(obj.clone()),
            None => Err(SpicyError::NameErr(id.to_owned())),
        }
    }

    pub fn has_var(&self, id: &str) -> Result<bool, SpicyError> {
        let vars = self
            .vars
            .read()
            .map_err(|_| SpicyError::ReadLockErr(id.to_owned()))?;
        Ok(vars.contains_key(id))
    }

    pub fn set_var(&self, id: &str, args: SpicyObj) -> SpicyResult<()> {
        let mut vars = self
            .vars
            .write()
            .map_err(|_| SpicyError::WriteLockErr(id.to_owned()))?;
        vars.insert(id.to_owned(), args);
        Ok(())
    }

    pub fn del_var(&self, id: &str) -> SpicyResult<SpicyObj> {
        let mut vars = self
            .vars
            .write()
            .map_err(|_| SpicyError::WriteLockErr(id.to_owned()))?;
        Ok(vars.remove(id).unwrap_or(SpicyObj::Null))
    }

    pub fn upsert_var(&self, id: &str, arg: &SpicyObj) -> SpicyResult<SpicyObj> {
        let mut vars = self
            .vars
            .write()
            .map_err(|_| SpicyError::WriteLockErr(id.to_owned()))?;
        let obj = match vars.get_mut(id) {
            Some(obj) => obj,
            None => {
                if arg.is_df() {
                    vars.insert(id.to_owned(), arg.clone());
                    return Ok(SpicyObj::I64(arg.size() as i64));
                } else {
                    return Err(SpicyError::Err(format!(
                        "the first upsert is required to be a dataframe, got {}",
                        arg.get_type_name()
                    )));
                }
            }
        };
        match obj.mut_df() {
            Ok(df) => match arg {
                SpicyObj::DataFrame(records) => {
                    df.extend(records)
                        .map_err(|e| SpicyError::Err(e.to_string()))?;
                    Ok(SpicyObj::I64(records.height() as i64))
                }
                SpicyObj::MixedList(list) => {
                    let df1 = convert_list_to_df(&list, &df)?;
                    df.extend(&df1)
                        .map_err(|e| SpicyError::Err(e.to_string()))?;
                    Ok(SpicyObj::I64(df1.height() as i64))
                }
                _ => Err(SpicyError::Err(format!(
                    "only allows to upsert (dataframe|list) to dataframe id, got {}",
                    arg.get_type_name()
                ))),
            },
            Err(_) => Err(SpicyError::Err(
                "only allows to upsert data to dataframe id".to_owned(),
            )),
        }
    }

    pub fn insert_var(&self, id: &str, args: &SpicyObj, by: &[&str]) -> SpicyResult<SpicyObj> {
        let mut vars = self
            .vars
            .write()
            .map_err(|_| SpicyError::WriteLockErr(id.to_owned()))?;
        let count: usize;
        let df = {
            let arg0 = match vars.get_mut(id) {
                Some(obj) => obj,
                None => {
                    if args.is_df() {
                        let df = args.df().unwrap().clone();
                        let df = df
                            .lazy()
                            .group_by(by)
                            .agg([col("*").last()])
                            .collect()
                            .map_err(|e| SpicyError::Err(e.to_string()))?;
                        count = df.height();
                        vars.insert(id.to_owned(), SpicyObj::DataFrame(df));
                        return Ok(SpicyObj::I64(count as i64));
                    } else {
                        return Err(SpicyError::Err(format!(
                            "the first insert is required to be a dataframe, got {}",
                            args.get_type_name()
                        )));
                    }
                }
            };
            match arg0.mut_df() {
                Ok(df) => {
                    count = df.height();
                    match args {
                        SpicyObj::DataFrame(records) => {
                            df.extend(records)
                                .map_err(|e| SpicyError::Err(e.to_string()))?;
                            df.clone()
                        }
                        SpicyObj::MixedList(list) => {
                            let records = convert_list_to_df(&list, &df)?;
                            df.extend(&records)
                                .map_err(|e| SpicyError::Err(e.to_string()))?;
                            df.clone()
                        }
                        _ => {
                            return Err(SpicyError::Err(format!(
                                "only allows to insert (dataframe|list) to dataframe id, got {}",
                                args.get_type_name()
                            )));
                        }
                    }
                }
                Err(_) => {
                    return Err(SpicyError::Err(
                        "only allows to insert data to dataframe id".to_owned(),
                    ));
                }
            }
        };
        let df = df
            .lazy()
            .group_by(by)
            .agg([col("*").last()])
            .collect()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let updated_count = df.height();
        vars.insert(id.to_owned(), SpicyObj::DataFrame(df));
        Ok(SpicyObj::I64(updated_count as i64 - count as i64))
    }

    pub fn list_vars(&self, pattern: &str) -> SpicyResult<DataFrame> {
        let vars: DataFrame = {
            let mut vars: Vec<String> = vec![];
            let mut displays: Vec<String> = vec![];
            let mut types: Vec<String> = vec![];
            let mut columns: Vec<String> = vec![];
            let mut is_built_in: Vec<bool> = vec![];
            let var_map = self
                .vars
                .read()
                .map_err(|e| SpicyError::ReadLockErr(e.to_string()))?;
            for (k, v) in var_map.iter() {
                if pattern.is_empty() || k.starts_with(pattern) {
                    vars.push(k.to_owned());
                    displays.push(v.to_string());
                    types.push(v.get_type_name());
                    if let SpicyObj::DataFrame(df) = v {
                        columns.push(
                            df.get_column_names()
                                .iter()
                                .map(|c| c.as_str())
                                .collect::<Vec<&str>>()
                                .join("|"),
                        );
                    } else {
                        columns.push("".to_string());
                    }
                    match v.fn_() {
                        Ok(func) => is_built_in.push(func.is_built_in_fn()),
                        Err(_) => is_built_in.push(false),
                    }
                }
            }
            DataFrame::new(
                vars.len(),
                vec![
                    Column::new("name".into(), vars),
                    Column::new("display".into(), displays),
                    Column::new("type".into(), types),
                    Column::new("columns".into(), columns),
                    Column::new("is_built_in".into(), is_built_in),
                ],
            )
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?
        };
        let par_df_vars = {
            let mut vars: Vec<String> = vec![];
            let mut displays: Vec<String> = vec![];
            let mut types: Vec<String> = vec![];
            let mut columns: Vec<String> = vec![];
            let mut is_built_in: Vec<bool> = vec![];
            let par_df_map = self
                .par_df
                .read()
                .map_err(|e| SpicyError::ReadLockErr(e.to_string()))?;
            for (k, v) in par_df_map.iter() {
                if pattern.is_empty() || k.starts_with(pattern) {
                    vars.push(k.to_owned());
                    displays.push(v.to_string());
                    types.push("par_df".to_string());
                    let lazy_df = v.scan_partition(0)?;
                    let cols = match lazy_df.collect() {
                        Ok(df) => df
                            .get_column_names()
                            .iter()
                            .map(|c| c.as_str())
                            .collect::<Vec<&str>>()
                            .join("|"),
                        Err(_) => "".to_string(),
                    };
                    columns.push(cols);
                    is_built_in.push(false);
                }
            }
            DataFrame::new(
                vars.len(),
                vec![
                    Column::new("name".into(), vars),
                    Column::new("display".into(), displays),
                    Column::new("type".into(), types),
                    Column::new("columns".into(), columns),
                    Column::new("is_built_in".into(), is_built_in),
                ],
            )
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?
        };
        vars.vstack(&par_df_vars)
            .map_err(|e| SpicyError::EvalErr(e.to_string()))
    }

    pub fn replay_q_msgs_log(
        &self,
        file: &str,
        start: i64,
        end: i64,
        table_names: &Vec<&str>,
    ) -> SpicyResult<SpicyObj> {
        if end == 0 {
            info!("no messages to replay");
            return Ok(SpicyObj::I64(0));
        }
        let mut skip = 8;
        let size_file = file.replace(".data", ".size");
        let size_bytes = fs::read(size_file).map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        let length = size_bytes.len() / 4;
        let ptr = size_bytes.as_ptr() as *const u32;
        let msgs_size = unsafe { core::slice::from_raw_parts(ptr, length) };
        let is_sub_all = table_names.is_empty();

        info!("total messages length on disk: {}", msgs_size.len());

        if start > msgs_size.len() as i64 {
            info!("already loaded all messages, no messages to replay");
            return Ok(SpicyObj::I64(0));
        }

        info!("replaying messages log from {} to {}", start + 1, end);

        let end = if end > msgs_size.len() as i64 {
            msgs_size.len() as i64
        } else {
            end
        };

        if start > 0 {
            for i in msgs_size.iter().take(start as usize) {
                skip += *i as usize;
            }
        }

        let tick_count = {
            *self
                .tick_count
                .read()
                .map_err(|e| SpicyError::EvalErr(e.to_string()))? as usize
        };

        let mut msgs_file = fs::OpenOptions::new()
            .read(true)
            .open(file)
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        msgs_file
            .seek(SeekFrom::Start(skip as u64))
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;

        let pb = ProgressBar::new((end - start) as u64);
        let style = ProgressStyle::with_template(
            "[{elapsed_precise}] {spinner:.green} [{bar:100.cyan/blue}] {pos:>7}/{len:>7}",
        )
        .unwrap();

        pb.set_style(style);

        let mut count = start as usize;
        for i in msgs_size.iter().take(end as usize).skip(start as usize) {
            let size = *i;
            let mut msg = vec![0u8; size as usize];

            msgs_file
                .read_exact(&mut msg)
                .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
            let table_name = read_q_table_name(&msg)?;

            if is_sub_all || table_names.contains(&table_name.as_str()) {
                if count >= tick_count {
                    let list = serde6::deserialize(&msg, &mut 0, false)?;
                    let res = self.eval(&mut Stack::default(), &list, "");
                    if res.is_err() {
                        let err = res.err().unwrap();
                        error!("failed to replay {} message, error: {}", i, err);
                        pb.finish_and_clear();
                        return Err(err); // TODO: remove this
                    }
                }
                count += 1;
            }
            if i % 100 == 0 {
                pb.inc(100);
            }
        }
        let replayed_count = count - tick_count;
        pb.finish();
        info!("replayed {} messages after filtering", replayed_count);
        Ok(SpicyObj::I64(replayed_count as i64))
    }

    pub fn replay_chili_msgs_log(
        &self,
        path: &str,
        start: i64,
        end: i64,
        start_time: i64,
        table_names: &Vec<&str>,
        eval: bool,
    ) -> SpicyResult<SpicyObj> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        if file
            .metadata()
            .map_err(|e| SpicyError::Err(e.to_string()))?
            .len()
            == 0
        {
            return Ok(SpicyObj::I64(0));
        }
        let mut file_header = [0u8; 8];
        file.read_exact(&mut file_header)
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        if file_header != [255, 0, 0, 0, 0, 0, 0, 0] {
            Err(SpicyError::Err("not a sequence file".to_string()))
        } else {
            let mut header = [0u8; 16];
            let mut read_msg_count = 0;
            let mut i = 0;
            let mut any = vec![];

            let pb = ProgressBar::new(end as u64);
            let style = ProgressStyle::with_template(
                "[{elapsed_precise}] {spinner:.green} [{bar:100.cyan/blue}] {pos:>7}/{len:>7}",
            )
            .unwrap();
            pb.set_style(style);

            while i < end {
                let res = file.read_exact(&mut header);
                if res.is_err() {
                    break;
                }
                let size = u64::from_le_bytes(header[..8].try_into().unwrap());
                let utc_time = u64::from_le_bytes(header[8..16].try_into().unwrap()) as i64;
                if (start_time > 0 && utc_time < start_time) || i < start {
                    file.seek(SeekFrom::Current(size as i64))
                        .map_err(|e| SpicyError::Err(e.to_string()))?;
                    i += 1;
                    pb.inc(1);
                    continue;
                }
                read_msg_count += 1;
                let mut buffer = vec![0u8; size as usize];
                file.read_exact(&mut buffer).map_err(|e| {
                    warn!("failed to read message at index {}: {}", i, e);
                    SpicyError::Err(e.to_string())
                })?;
                let list = serde9::deserialize(&buffer, &mut 0)?;
                if eval {
                    if table_names.is_empty()
                        || table_names.contains(
                            &list
                                .as_vec()?
                                .get(1)
                                .unwrap_or(&SpicyObj::String("".to_string()))
                                .str()?,
                        )
                    {
                        let res = self.eval(&mut Stack::default(), &list, "");
                        if res.is_err() {
                            let err = res.err().unwrap();
                            error!(
                                "failed to replay {} message, error: {}",
                                read_msg_count, err
                            );
                            return Err(err);
                        }
                    }
                } else {
                    any.push(list);
                }
                i += 1;
                pb.inc(1);
            }
            pb.set_length(i as u64);
            pb.finish();
            if eval {
                Ok(SpicyObj::I64(read_msg_count))
            } else {
                Ok(SpicyObj::MixedList(any))
            }
        }
    }

    pub fn open_handle(&self, uri: &str, h: i64) -> SpicyResult<SpicyObj> {
        let mut callback = None;
        let uri = if h > 0 {
            let handles = self
                .handle
                .read()
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            if handles.contains_key(&h) {
                let handle = handles.get(&h).unwrap();
                callback = handle.on_disconnected.clone();
                handle.uri.clone()
            } else {
                return Err(SpicyError::InvalidHandleErr(h));
            }
        } else {
            uri.to_owned()
        };

        let err = SpicyError::EvalErr(format!(
            "invalid uri format, expected\n\
            - 'q://host:port[:user:password]'\n\
            - 'chili://host:port[:user:password]'\n\
            - 'file://path'\n\
            got '{}'",
            uri
        ));

        let (ipc_type, socket, version) = match uri.split_once("://") {
            Some((schema, path)) => {
                if schema == "q" {
                    (IpcType::Q, path, 6)
                } else if schema == "chili" {
                    (IpcType::Chili, path, 9)
                } else if schema == "file" {
                    let mut file = fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .truncate(false)
                        .open(path)
                        .map_err(|e| SpicyError::Err(e.to_string()))?;
                    let metadata = file
                        .metadata()
                        .map_err(|e| SpicyError::Err(e.to_string()))?;
                    let conn_type = if metadata.len() == 0 {
                        ConnType::New
                    } else if metadata.len() < 8 {
                        ConnType::File
                    } else {
                        let mut header = [0u8; 4];
                        file.read_exact(&mut header).map_err(|e| {
                            SpicyError::Err(format!("failed to read header, error: {}", e))
                        })?;
                        if [255, 0, 0, 0] == header {
                            ConnType::Sequence
                        } else {
                            ConnType::File
                        }
                    };
                    file.seek(SeekFrom::End(0)).map_err(|e| {
                        SpicyError::Err(format!("failed to seek to end of file, error: {}", e))
                    })?;
                    let h = self.set_handle(
                        Some(Box::new(file)),
                        &format!("file://{}", path),
                        &uri,
                        false,
                        IpcType::Chili,
                        conn_type,
                        0,
                    )?;
                    return Ok(h);
                } else {
                    return Err(err);
                }
            }
            None => return Err(err),
        };

        let (host, port, user, password) = unpack_socket(socket)?;

        let mut stream = match TcpStream::connect(format!("{}:{}", host, port)) {
            Ok(s) => s,
            Err(e) => return Err(SpicyError::Err(e.to_string())),
        };
        stream.set_nodelay(true).unwrap();

        let remote_version = send_auth(&mut stream, &user, &password, version)?;

        if remote_version != version {
            stream.shutdown(std::net::Shutdown::Both).unwrap();
            return Err(SpicyError::Err(format!(
                "mismatched version, remote: {}, local: {}, use `q:// for q process",
                remote_version, version
            )));
        }

        let is_local = host.starts_with("localhost") | host.starts_with("127.0.0.1");
        let h = self.set_handle(
            Some(Box::new(stream)),
            &format!("{}://{}:{}", ipc_type, host, port),
            &uri,
            is_local,
            ipc_type,
            ConnType::Outgoing,
            h,
        )?;
        if let Some(callback) = callback {
            self.set_callback(h.i64().unwrap(), callback)?;
        }
        Ok(h)
    }

    pub fn close_handle(&self, handle_num: &i64) -> SpicyResult<SpicyObj> {
        let mut handle = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        handle.shift_remove(handle_num);
        Ok(SpicyObj::Null)
    }

    pub fn list_handle(&self) -> SpicyResult<DataFrame> {
        let handles = self.handle.read().unwrap();
        let len = handles.len();
        let mut num = Vec::with_capacity(len);
        let mut socket = Vec::with_capacity(len);
        let mut conn_type = Vec::with_capacity(len);
        let mut ipc_type = Vec::with_capacity(len);
        let mut is_local = Vec::with_capacity(len);
        let mut on_disconnected = Vec::with_capacity(len);
        for (k, v) in handles.iter() {
            num.push(*k);
            socket.push(v.socket.clone());
            conn_type.push(format!("{:?}", v.conn_type));
            ipc_type.push(format!("{:?}", v.ipc_type));
            is_local.push(v.is_local);
            on_disconnected.push(v.on_disconnected.clone().unwrap_or_default());
        }
        DataFrame::new(
            len,
            vec![
                Column::new("num".into(), num),
                Column::new("socket".into(), socket),
                Column::new("conn_type".into(), conn_type),
                Column::new("ipc_type".into(), ipc_type),
                Column::new("is_local".into(), is_local),
                Column::new("on_disconnected".into(), on_disconnected),
            ],
        )
        .map_err(|e| SpicyError::EvalErr(e.to_string()))
    }

    pub fn disconnect_handle(&self, handle_num: &i64) -> SpicyResult<SpicyObj> {
        let mut handle = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        handle.get_mut(handle_num).unwrap().conn_type = ConnType::Disconnected;
        Ok(SpicyObj::Null)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn set_handle(
        &self,
        rw: Option<Box<dyn ReadWrite>>,
        socket: &str,
        uri: &str,
        is_local: bool,
        ipc_type: IpcType,
        conn_type: ConnType,
        h: i64,
    ) -> SpicyResult<SpicyObj> {
        let mut handle = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let h = if handle.contains_key(&h) {
            h
        } else {
            1 + handle.keys().max().copied().unwrap_or(3)
        };
        handle.insert(
            h,
            Handle {
                rw,
                socket: socket.to_owned(),
                uri: uri.to_owned(),
                is_local,
                ipc_type,
                conn_type,
                on_disconnected: None,
            },
        );
        Ok(SpicyObj::I64(h))
    }

    pub fn set_callback(&self, h: &i64, callback: String) -> SpicyResult<()> {
        let mut handle = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        if let Some(handle) = handle.get_mut(h) {
            handle.on_disconnected = Some(callback);
        } else {
            return Err(SpicyError::InvalidHandleErr(*h));
        }
        Ok(())
    }

    pub fn get_callback(&self, h: &i64) -> SpicyResult<String> {
        let handle = self
            .handle
            .read()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        if let Some(handle) = handle.get(h) {
            Ok(handle.on_disconnected.clone().unwrap_or_default())
        } else {
            Err(SpicyError::InvalidHandleErr(*h))
        }
    }

    pub fn handle_publisher(&self, h: &i64) -> SpicyResult<()> {
        let mut handles = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let publisher = handles.shift_remove(h);

        match publisher {
            Some(handle) => {
                let arc_self = self
                    .arc_self
                    .read()
                    .map_err(|e| SpicyError::Err(e.to_string()))?;

                let arc_self = Arc::clone(arc_self.as_ref().unwrap());
                if handle.conn_type != ConnType::Outgoing {
                    return Err(SpicyError::Err(format!(
                        "requires an outgoing connection for subscribing, got {:?}",
                        handle.conn_type
                    )));
                }
                let h = *h;
                handles.insert(
                    h,
                    Handle {
                        rw: None,
                        socket: handle.socket,
                        uri: handle.uri,
                        is_local: handle.is_local,
                        ipc_type: handle.ipc_type,
                        conn_type: ConnType::Subscribing,
                        on_disconnected: handle.on_disconnected,
                    },
                );
                let user = self.user.clone();
                if handle.ipc_type == IpcType::Q {
                    thread::spawn(move || {
                        handle_q_conn(&mut handle.rw.unwrap(), handle.is_local, h, arc_self, &user);
                    });
                } else if handle.ipc_type == IpcType::Chili {
                    thread::spawn(move || {
                        handle_chili_conn(
                            &mut handle.rw.unwrap(),
                            handle.is_local,
                            h,
                            arc_self,
                            &user,
                        );
                    });
                } else {
                    return Err(SpicyError::EvalErr(format!(
                        "invalid ipc type: {:?}, requires q or chili",
                        handle.ipc_type
                    )));
                }
                Ok(())
            }
            None => Err(SpicyError::InvalidHandleErr(*h)),
        }
    }

    pub fn sync(&self, h: &i64, msg: &SpicyObj) -> SpicyResult<SpicyObj> {
        let mut handle = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        match handle.get_mut(h) {
            Some(Handle {
                rw: Some(rw),
                is_local,
                ipc_type,
                conn_type,
                ..
            }) => {
                if *conn_type == ConnType::Outgoing {
                    match msg {
                        SpicyObj::Symbol(_) | SpicyObj::String(_) | SpicyObj::MixedList(_) => {
                            if *ipc_type == IpcType::Q {
                                let v8 = serde6::serialize(msg)?;
                                let v8 = if !*is_local { serde6::compress(v8) } else { v8 };
                                if let Err(e) = utils::write_q_ipc_msg(rw, &v8, MessageType::Sync) {
                                    self.disconnect_handle(h)?;
                                    return Err(SpicyError::Err(e.to_string()));
                                }
                                // read response
                                let mut header = [0u8; 8];
                                rw.read_exact(&mut header)
                                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                                let (message_type, len, compression_mode) =
                                    utils::decode_header6(&header);
                                let any = read_q_msg(rw, len - 8, compression_mode)?;
                                if message_type == MessageType::Response {
                                    Ok(any)
                                } else {
                                    Err(SpicyError::EvalErr(format!(
                                        "expected response, got {:?}",
                                        message_type
                                    )))
                                }
                            } else {
                                let v8 = serde9::serialize(msg, !*is_local)?;
                                if let Err(e) =
                                    utils::write_chili_ipc_msg(rw, &v8, MessageType::Sync)
                                {
                                    self.disconnect_handle(h)?;
                                    return Err(SpicyError::Err(e.to_string()));
                                }
                                let mut header = [0u8; 16];
                                rw.read_exact(&mut header)
                                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                                let (message_type, len) = utils::decode_header9(&header);
                                let obj = read_chili_ipc_msg(rw, len)?;
                                if message_type == MessageType::Response {
                                    Ok(obj)
                                } else {
                                    Err(SpicyError::EvalErr(format!(
                                        "expected response, got {:?}",
                                        message_type
                                    )))
                                }
                            }
                        }
                        _ => Err(SpicyError::MismatchedTypeErr(
                            "sym|str|mixedList".to_owned(),
                            msg.get_type_name(),
                        )),
                    }
                } else if *conn_type == ConnType::New {
                    match msg {
                        SpicyObj::Symbol(s) | SpicyObj::String(s) => {
                            writeln!(rw, "{}", s).map_err(|e| SpicyError::Err(e.to_string()))?;
                            *conn_type = ConnType::File;
                            Ok(SpicyObj::I64(s.len() as i64))
                        }
                        SpicyObj::MixedList(_) => {
                            rw.write_all(&[255, 0, 0, 0, 0, 0, 0, 0])
                                .map_err(|e| SpicyError::Err(e.to_string()))?;
                            let bytes_vec = serde9::serialize(msg, false)?;
                            let total_len = bytes_vec.iter().map(|v| v.len()).sum::<usize>();
                            rw.write(total_len.to_le_bytes().as_slice())
                                .map_err(|e| SpicyError::Err(e.to_string()))?;
                            rw.write(
                                &(SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_nanos() as u64)
                                    .to_le_bytes(),
                            )
                            .map_err(|e| SpicyError::Err(e.to_string()))?;
                            for bytes in bytes_vec {
                                rw.write_all(&bytes)
                                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                            }
                            *conn_type = ConnType::Sequence;
                            Ok(SpicyObj::I64(total_len as i64))
                        }
                        _ => Err(SpicyError::MismatchedTypeErr(
                            "sym|str".to_owned(),
                            msg.get_type_name(),
                        )),
                    }
                } else if *conn_type == ConnType::File {
                    match msg {
                        SpicyObj::Symbol(s) | SpicyObj::String(s) => {
                            writeln!(rw, "{}", s).map_err(|e| SpicyError::Err(e.to_string()))?;
                            Ok(SpicyObj::I64(s.len() as i64))
                        }
                        _ => Err(SpicyError::MismatchedTypeErr(
                            "sym|str".to_owned(),
                            msg.get_type_name(),
                        )),
                    }
                } else if *conn_type == ConnType::Sequence {
                    match msg {
                        SpicyObj::MixedList(_) => {
                            let bytes_vec = serde9::serialize(msg, false)?;
                            let total_len = bytes_vec.iter().map(|v| v.len()).sum::<usize>();
                            rw.write(total_len.to_le_bytes().as_slice())
                                .map_err(|e| SpicyError::Err(e.to_string()))?;
                            rw.write(
                                &(SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap()
                                    .as_nanos() as u64)
                                    .to_le_bytes(),
                            )
                            .map_err(|e| SpicyError::Err(e.to_string()))?;
                            for bytes in bytes_vec {
                                rw.write_all(&bytes)
                                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                            }
                            *conn_type = ConnType::Sequence;
                            Ok(SpicyObj::I64(total_len as i64))
                        }
                        _ => Err(SpicyError::MismatchedTypeErr(
                            "mixedList".to_owned(),
                            msg.get_type_name(),
                        )),
                    }
                } else {
                    Err(SpicyError::EvalErr(format!(
                        "cannot sync for {:?} handle",
                        conn_type
                    )))
                }
            }
            _ => Err(SpicyError::InvalidHandleErr(*h)),
        }
    }

    pub fn add_subscriber(&self, topic: &str, h: i64) -> SpicyResult<()> {
        let mut topic_map = self
            .topic_map
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        topic_map.entry(topic.to_owned()).or_insert(vec![]).push(h);
        Ok(())
    }

    pub fn remove_subscriber(&self, topic: &str, h: i64) -> SpicyResult<()> {
        let mut topic_map = self
            .topic_map
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        topic_map
            .entry(topic.to_owned())
            .or_insert(vec![])
            .retain(|&x| x != h);
        Ok(())
    }

    pub fn publish(&self, table: &str, bytes: &[Vec<u8>]) -> SpicyResult<()> {
        let mut topic_map = self
            .topic_map
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let subscribers = match topic_map.get(table) {
            Some(subscribers) => subscribers.clone(),
            None => {
                debug!("no subscribers for table '{}', skip publish", table);
                return Ok(());
            }
        };

        if subscribers.is_empty() {
            debug!("no subscribers for table '{}', skip publish", table);
            return Ok(());
        }

        let mut handle = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        for subscriber in subscribers {
            if let Some(v) = handle.get_mut(&subscriber) {
                if v.conn_type == ConnType::Disconnected {
                    continue;
                }
                match &mut v.rw {
                    Some(rw) => match crate::write_chili_ipc_msg(rw, bytes, MessageType::Async) {
                        Ok(_) => (),
                        Err(e) => {
                            warn!(
                                "failed to write to handle {} - err {}, disconnecting...",
                                subscriber, e
                            );
                            v.conn_type = ConnType::Disconnected;
                        }
                    },
                    None => {
                        warn!("handle {} is disconnected", subscriber);
                        v.conn_type = ConnType::Disconnected;
                    }
                }
            } else {
                warn!(
                    "subscriber {} is not found, removing from topic map",
                    subscriber
                );
                topic_map
                    .get_mut(table)
                    .unwrap()
                    .retain(|x| *x != subscriber);
            }
        }
        Ok(())
    }

    pub fn signal_eod(&self, args: &SpicyObj) -> SpicyResult<()> {
        let handles: Vec<i64> = {
            let handle = self
                .handle
                .read()
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            handle
                .iter()
                .filter(|(_, v)| v.conn_type == ConnType::Publishing)
                .map(|(k, _)| *k)
                .collect()
        };
        for h in handles {
            if let Err(e) = self.sync(&h, args) {
                warn!(
                    "failed to signal EOD to handle {} - err {}, disconnecting...",
                    h, e
                );
                self.disconnect_handle(&h)?;
            }
        }
        Ok(())
    }

    pub fn handle_subscriber(&self, h: &i64) -> SpicyResult<()> {
        let mut handles = self
            .handle
            .write()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let handle = handles.get_mut(h).ok_or(SpicyError::InvalidHandleErr(*h))?;

        if handle.conn_type == ConnType::Publishing {
            return Ok(());
        }

        if handle.conn_type == ConnType::Incoming {
            handle.conn_type = ConnType::Publishing;
            Ok(())
        } else {
            Err(SpicyError::Err(format!(
                "requires an incoming connection for publishing, got {:?}",
                handle.conn_type
            )))
        }
    }

    pub fn list_topic_map(&self) -> SpicyResult<DataFrame> {
        let topic_map = self
            .topic_map
            .read()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        let mut topics = Vec::new();
        let mut subscribers: Vec<i64> = Vec::new();
        let mut offsets: Vec<i32> = Vec::with_capacity(topic_map.len() + 1);
        offsets.push(0);
        let mut offset = 0;
        for (topic, handles) in topic_map.iter() {
            topics.push(topic.clone());
            subscribers.extend(handles);
            offset += handles.len();
            offsets.push(offset as i32);
        }
        let series = Series::from_arrow(
            "subscribers".into(),
            ListArray::new(
                ArrowDataType::List(Box::new(ArrowField::new(
                    "subscribers".into(),
                    ArrowDataType::Int64,
                    true,
                ))),
                OffsetsBuffer::<i32>::try_from(offsets).unwrap(),
                Int64Array::from_vec(subscribers).boxed(),
                None,
            )
            .boxed(),
        )
        .unwrap();
        DataFrame::new(
            topics.len(),
            vec![Column::new("topic".into(), topics), series.into()],
        )
        .map_err(|e| SpicyError::EvalErr(e.to_string()))
    }

    pub fn get_source(&self, index: usize) -> SpicyResult<(String, String)> {
        let source = self
            .source
            .read()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        Ok(source
            .get(index)
            .cloned()
            .unwrap_or(("".to_owned(), "".to_owned())))
    }

    pub fn set_source(&self, path: &str, src: &str) -> SpicyResult<usize> {
        // Idempotent: if (path, src) already exists in the source registry,
        // return the existing source_id rather than appending a duplicate.
        // Required for parse cache concurrency correctness — without this,
        // two threads racing on the same uncached query both append to the
        // source vec, producing distinct source_ids for identical content
        // and bloating the source registry under concurrent Python load.
        // O(n) scan is fine because the registry is small (typically <1000
        // entries) and writes are off the hot path.
        let mut source = self
            .source
            .write()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        if let Some(idx) = source.iter().position(|(p, s)| p == path && s == src) {
            return Ok(idx);
        }
        source.push((path.to_owned(), src.to_owned()));
        Ok(source.len() - 1)
    }

    pub fn parse(&self, path: &str, source: &str) -> Result<Vec<AstNode>, SpicyError> {
        // Look up by (path, source). On cache hit, return a clone of the
        // cached AST WITHOUT calling set_source (the cached AST already
        // embeds source positions for the original source_id, which is
        // still alive in self.source — sources are append-only so once
        // assigned a source_id is valid for the lifetime of the engine).
        //
        // Double-check pattern: probe under the lock, miss → drop lock,
        // do the slow parse outside the lock, then re-acquire to insert.
        // This avoids holding the parse_cache mutex across the chumsky
        // parse call (which can be milliseconds long for big queries).

        let cache_key = (path.to_string(), source.to_string());

        // Fast path: cache hit
        if let Ok(mut cache) = self.parse_cache.lock() {
            if let Some(ast) = cache.get(&cache_key) {
                return Ok((**ast).clone());
            }
        }
        // (lock dropped here)

        // Slow path: parse and insert
        let source_id = if !path.is_empty() {
            self.set_source(path, source).unwrap()
        } else {
            0
        };
        let parsed = if path.is_empty() {
            let path = if self.repl_lang == Language::Chili {
                "repl.chi"
            } else {
                "repl.pep"
            };
            parse(source, source_id, path)?
        } else {
            parse(source, source_id, path)?
        };

        let arc = Arc::new(parsed.clone());
        if let Ok(mut cache) = self.parse_cache.lock() {
            cache.put(cache_key, arc);
        }
        Ok(parsed)
    }

    /// Returns the current parse cache size (mostly for tests / observability).
    pub fn parse_cache_len(&self) -> usize {
        self.parse_cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    pub fn parse_raw_fn(&self, fn_body: &str, lang: Language) -> Result<Vec<AstNode>, SpicyError> {
        if lang == Language::Chili {
            parse(fn_body, 0, "repl.chi")
        } else {
            parse(fn_body, 0, "repl.pep")
        }
    }

    pub fn eval_ast(
        &self,
        nodes: Vec<AstNode>,
        src_path: &str,
        src: &str,
    ) -> SpicyResult<SpicyObj> {
        let src_path = if src_path.is_empty() {
            None
        } else {
            Some(src_path.to_owned())
        };
        let mut stack = Stack::new(src_path, 0, 0, "");
        let mut obj = SpicyObj::Null;
        for node in nodes {
            obj = eval_by_node(self, &mut stack, &node, src, None)?;
        }
        Ok(obj)
    }

    pub fn import_source_path(&self, relative_src_path: &str, path: &str) -> SpicyResult<SpicyObj> {
        #[cfg(target_os = "windows")]
        let full_path = {
            let is_windows_full_path = path.chars().nth(1).unwrap_or(' ') == ':';
            if !path.starts_with(".") && !is_windows_full_path {
                return Err(SpicyError::EvalErr(format!(
                    "invalid path '{}', expected absolute or relative path, start with '/' or '.'",
                    path
                )));
            }
            let path = &path.replace("/", "\\");
            if relative_src_path.is_empty() {
                fs::canonicalize(path).map_err(|e| {
                    SpicyError::EvalErr(format!("failed to locate '{}', {}", path, e))
                })?
            } else {
                let relative_path = PathBuf::from(relative_src_path);
                match relative_path.parent() {
                    Some(base_path) if path.starts_with(".") => {
                        let full_path = base_path.join(path);
                        fs::canonicalize(&full_path).map_err(|e| {
                            SpicyError::EvalErr(format!(
                                "failed to locate '{}', {}",
                                full_path.display(),
                                e
                            ))
                        })?
                    }
                    _ => fs::canonicalize(path).map_err(|e| {
                        SpicyError::EvalErr(format!("failed to locate '{}', {}", path, e))
                    })?,
                }
            }
        };

        #[cfg(not(target_os = "windows"))]
        let full_path = {
            if !path.starts_with("/") && !path.starts_with(".") {
                return Err(SpicyError::EvalErr(format!(
                    "invalid path '{}', expected absolute or relative path, start with '/' or '.'",
                    path
                )));
            }
            if relative_src_path.is_empty() {
                fs::canonicalize(path).map_err(|e| {
                    SpicyError::EvalErr(format!("failed to locate '{}', {}", path, e))
                })?
            } else {
                let relative_path = PathBuf::from(relative_src_path);
                match relative_path.parent() {
                    Some(base_path) if path.starts_with(".") => {
                        let full_path = base_path.join(path);
                        fs::canonicalize(&full_path).map_err(|e| {
                            SpicyError::EvalErr(format!(
                                "failed to locate '{}', {}",
                                full_path.display(),
                                e
                            ))
                        })?
                    }
                    _ => fs::canonicalize(path).map_err(|e| {
                        SpicyError::EvalErr(format!("failed to locate '{}', {}", path, e))
                    })?,
                }
            }
        };

        let full_path = full_path.to_string_lossy().to_string();

        let src = fs::read_to_string(&full_path)
            .map_err(|e| SpicyError::EvalErr(format!("failed to read '{}', {}", &full_path, e)))?;

        if self
            .source
            .read()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?
            .iter()
            .any(|(p, s)| p == &full_path && s == &src)
        {
            info!("source '{}' already loaded", full_path);
            return Ok(SpicyObj::Null);
        }

        debug!("loading '{}'", full_path);
        let nodes = self
            .parse(&full_path, &src)
            .map_err(|e| SpicyError::Err(format!("failed to parse '{}'\n{}", full_path, e)))?;

        self.eval_ast(nodes, &full_path, &src)
            .map_err(|e| SpicyError::EvalErr(format!("'{}'\n{}", full_path, e)))
    }

    pub fn eval(
        &self,
        stack: &mut Stack,
        args: &SpicyObj,
        src_path: &str,
    ) -> SpicyResult<SpicyObj> {
        match args {
            SpicyObj::String(source) => {
                let nodes = self
                    .parse("", source)
                    .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
                self.eval_ast(nodes, src_path, source)
            }
            SpicyObj::MixedList(_) if args.size() == 0 => Ok(SpicyObj::Null),
            SpicyObj::Series(_) if args.is_syms() || args.is_str_or_strs() => {
                let args = args.as_vec().unwrap();
                eval_op(self, stack, &[&SpicyObj::MixedList(args)])
            }
            SpicyObj::MixedList(_) | SpicyObj::Symbol(_) => eval_op(self, stack, &[args]),
            _ => Err(SpicyError::EvalErr(format!(
                "Unable to eval '{}'",
                args.get_type_name()
            ))),
        }
    }

    pub fn fn_call(&self, func: &str, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
        let func = self.get_var(func)?;
        let mut stack = Stack::new(None, 0, 0, "");
        match func {
            SpicyObj::Fn(f) => eval_fn_call(self, &mut stack, &f, &args.to_vec()),
            _ => Err(SpicyError::EvalErr(format!(
                "Not able to call '{}'",
                func.get_type_name()
            ))),
        }
    }

    pub fn get_par_df(&self, name: &str) -> SpicyResult<PartitionedDataFrame> {
        let par_df = self
            .par_df
            .read()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        match par_df.get(name) {
            Some(p) => Ok(p.clone()),
            None => Err(SpicyError::Err(format!(
                "partitioned table '{}' not found",
                name
            ))),
        }
    }

    pub fn clear_par_df(&self) -> SpicyResult<()> {
        let mut par_df = self
            .par_df
            .write()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        par_df.clear();
        Ok(())
    }

    pub fn load_par_df(&self, path: &str) -> SpicyResult<()> {
        // Two-phase load:
        //   1. (outside lock) enumerate top-level table entries; build each
        //      PartitionedDataFrame in parallel via rayon; read the schema
        //      sentinel for each table at the same time so queries that miss
        //      a partition never re-open the schema file (Proposal J).
        //   2. (inside lock) acquire the par_df write lock exactly once and
        //      extend the map. This shrinks the lock window from "entire
        //      directory traversal" to "one atomic HashMap::extend", so
        //      concurrent readers are no longer blocked during reload.

        // Phase 1a: collect top-level table entries. We eagerly materialise
        // into Vec<(PathBuf, String, bool)> because DirEntry is not Send on
        // all platforms, so we cannot pass the iterator directly to rayon.
        let paths = match fs::read_dir(path) {
            Ok(p) => p,
            Err(e) => {
                return Err(SpicyError::EvalErr(format!("OS err: {}", e)));
            }
        };
        let entries: Vec<(PathBuf, String, bool)> = paths
            .filter_map(|e| match e {
                Ok(entry) => {
                    let is_file = entry.metadata().ok().map(|m| m.is_file()).unwrap_or(false);
                    let name = entry.file_name().to_string_lossy().to_string();
                    Some((entry.path(), name, is_file))
                }
                Err(e) => {
                    eprintln!("OS err: {}", e);
                    None
                }
            })
            .collect();

        // Phase 1b: build each PartitionedDataFrame in parallel. Each table
        // is independent so rayon `par_iter` is safe and gives roughly Nx
        // speedup on multi-table HDBs (Proposal C).
        let new_entries: Vec<(String, PartitionedDataFrame)> = entries
            .par_iter()
            .filter_map(|(table_path, table_name, is_file)| {
                Self::build_par_df_entry(table_path, table_name.clone(), *is_file)
            })
            .collect();

        // Phase 2: acquire the write lock exactly once and extend. This is
        // the only place the lock is held during load_par_df, and its
        // duration is bounded by HashMap::insert, not filesystem I/O.
        let mut par_df = self
            .par_df
            .write()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        for (k, v) in new_entries {
            par_df.insert(k, v);
        }
        Ok(())
    }

    /// Build a single `PartitionedDataFrame` entry for one top-level entry
    /// under the HDB root. Returns `Some((table_name, par_df))` on success,
    /// `None` when the entry should be skipped (empty directory, invalid
    /// filenames, etc.). Runs entirely outside any shared lock — safe to
    /// invoke from rayon workers.
    fn build_par_df_entry(
        table_path: &PathBuf,
        table_name: String,
        is_file: bool,
    ) -> Option<(String, PartitionedDataFrame)> {
        // Top-level file → single table.
        if is_file {
            info!("loading single dataframe '{}'...", table_name);
            return Some((
                table_name.clone(),
                PartitionedDataFrame {
                    name: table_name,
                    df_type: DFType::Single,
                    path: table_path.to_string_lossy().to_string(),
                    pars: vec![],
                    empty_schema: None,
                },
            ));
        }

        // Top-level directory → partitioned table.
        let dir = match fs::read_dir(table_path) {
            Ok(d) => d,
            Err(e) => {
                error!("failed to read table dir {:?}: {}", table_path, e);
                return None;
            }
        };
        let files: Vec<DirEntry> = dir
            .into_iter()
            .flatten()
            .filter(|p| p.metadata().is_ok() && p.metadata().unwrap().is_file())
            .collect();
        let filename_len = files.iter().map(|f| f.file_name().len()).max().unwrap_or(0);
        if filename_len == 0 {
            error!(
                "skip partitioned dataframe '{}' because no files found",
                table_name
            );
            return None;
        }
        let df_type = if filename_len >= 13 {
            DFType::ByDate
        } else {
            DFType::ByYear
        };

        info!(
            "loading dataframe '{}' partitioned by '{}'...",
            table_name, df_type
        );

        let mut par_vec: Vec<i32> = Vec::new();
        for p in &files {
            let file_name = p.file_name().to_string_lossy().to_string();
            if file_name == "schema" {
                continue;
            }
            if df_type == DFType::ByDate {
                if file_name.len() < 13 {
                    error!(
                        "partitioned by date shall match filename(0000.00.00_*), skip {:?}..",
                        p.file_name()
                    );
                    continue;
                }
                let date = SpicyObj::parse_date(&file_name[..10]);
                match date {
                    Ok(d) => {
                        let d = d.to_i64().unwrap();
                        if d > 3999 && !par_vec.contains(&(d as i32)) {
                            par_vec.push(d as i32)
                        }
                    }
                    Err(e) => eprintln!("{}", e),
                }
            } else {
                if file_name.len() < 7 {
                    error!(
                        "partitioned by year shall match filename(0000_00), skip {:?}..",
                        p.file_name()
                    );
                    continue;
                }
                let year = file_name[..4].parse::<i32>();
                match year {
                    Ok(y) => {
                        if y <= 3999 && !par_vec.contains(&y) {
                            par_vec.push(y)
                        }
                    }
                    Err(e) => {
                        error!("failed to parse '{}' - err {}", file_name, e)
                    }
                }
            }
        }
        if par_vec.is_empty() {
            return None;
        }
        // `fs::read_dir` yields entries in filesystem-dependent order
        // (unsorted on macOS APFS and many Linux FSes). `PartitionedDataFrame`
        // uses `slice::binary_search` for scan_partition / scan_partition_by_range,
        // which has undefined behavior on unsorted input. Sort here so
        // partition lookups are deterministic.
        par_vec.sort_unstable();

        // Proposal J: read the schema sentinel parquet file once at load
        // time and cache it as an empty DataFrame. Queries that hit a
        // missing partition return `empty_schema.clone().lazy()` instead
        // of re-scanning the sentinel file on every miss.
        let empty_schema = {
            let mut schema_path = table_path.clone();
            schema_path.push("schema");
            match std::fs::File::open(&schema_path) {
                Ok(f) => match ParquetReader::new(f).finish() {
                    Ok(df) => Some(Arc::new(df)),
                    Err(e) => {
                        warn!(
                            "failed to pre-read schema sentinel for {}: {} — will fall back to per-miss scan",
                            table_name, e
                        );
                        None
                    }
                },
                Err(_) => None, // no schema file — leave as None; scans fall back
            }
        };

        Some((
            table_name.clone(),
            PartitionedDataFrame {
                name: table_name,
                df_type,
                path: table_path.to_string_lossy().to_string(),
                pars: par_vec,
                empty_schema,
            },
        ))
    }

    pub fn tick(&self, inc: i64) -> SpicyResult<SpicyObj> {
        let mut tick_count = self
            .tick_count
            .write()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        *tick_count += inc;
        Ok(SpicyObj::I64(*tick_count))
    }

    pub fn get_tick_count(&self) -> i64 {
        *self.tick_count.read().unwrap()
    }

    pub fn get_table_names(&self, start_with: &str) -> SpicyResult<SpicyObj> {
        let vars = self
            .vars
            .read()
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        let mut table_names = Vec::new();
        for (k, obj) in vars.iter() {
            if k.starts_with(start_with) && obj.is_df() {
                table_names.push(k.clone());
            }
        }
        Ok(SpicyObj::Series(Series::new(
            "table_names".into(),
            table_names,
        )))
    }

    pub fn execute_jobs(&self) {
        let mut active_jobs: HashMap<i64, Job> = HashMap::new();
        {
            let jobs = self.job.read().unwrap();
            for (id, job) in jobs.iter() {
                if job.is_active && job::get_local_now_ns() >= job.next_run_time {
                    active_jobs.insert(*id, job.clone());
                }
            }
        }

        if !active_jobs.is_empty() {
            for (id, job) in active_jobs.iter_mut() {
                let src_path = if self.repl_lang == Language::Chili {
                    format!("job{}.chi", id)
                } else {
                    format!("job{}.pep", id)
                };
                let obj = self.eval(
                    &mut Stack::new(None, 0, 0, ""),
                    &SpicyObj::String(self.repl_lang.format_call(&job.fn_name, &[])),
                    &src_path,
                );
                if let Err(e) = obj {
                    error!(
                        "failed to execute job id '{}' , fn_name '{}', err - {}\n",
                        id, job.fn_name, e
                    );
                };
                if job.next_run_time + job.interval < job.end_time {
                    job.next_run_time += job.interval;
                } else {
                    job.is_active = false;
                }
                job.last_run_time = Some(job::get_local_now_ns());
            }

            self.job.write().unwrap().extend(active_jobs);
        }
    }

    pub fn add_job(&self, job: Job) -> i64 {
        let mut jobs = self.job.write().unwrap();
        let id = jobs.len() as i64 + 1;
        jobs.insert(id, job);
        id
    }

    pub fn list_job(&self) -> SpicyResult<DataFrame> {
        let jobs = self.job.read().unwrap();
        let mut id = vec![];
        let mut fn_name = vec![];
        let mut start_time = vec![];
        let mut end_time = vec![];
        let mut interval = vec![];
        let mut last_run_time = vec![];
        let mut next_run_time = vec![];
        let mut is_active = vec![];
        let mut description = vec![];
        for (i, job) in jobs.iter() {
            id.push(*i);
            fn_name.push(job.fn_name.clone());
            start_time.push(job.start_time);
            end_time.push(job.end_time);
            interval.push(job.interval);
            last_run_time.push(job.last_run_time);
            next_run_time.push(job.next_run_time);
            is_active.push(job.is_active);
            description.push(job.description.clone());
        }
        let df = DataFrame::new(
            jobs.len(),
            vec![
                Column::new("id".into(), id),
                Column::new("fn_name".into(), fn_name),
                Column::new("start_time".into(), start_time)
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap(),
                Column::new("end_time".into(), end_time)
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap(),
                Column::new("interval".into(), interval)
                    .cast(&DataType::Duration(TimeUnit::Nanoseconds))
                    .unwrap(),
                Column::new("last_run_time".into(), last_run_time)
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap(),
                Column::new("next_run_time".into(), next_run_time)
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap(),
                Column::new("is_active".into(), is_active),
                Column::new("description".into(), description),
            ],
        )
        .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        Ok(df)
    }

    pub fn set_job_status(&self, id: i64, is_active: bool) -> SpicyResult<i64> {
        let mut jobs = self.job.write().unwrap();
        let job = jobs.get_mut(&id).unwrap();
        job.is_active = is_active;
        Ok(id)
    }

    pub fn set_job_status_by_pattern(&self, pattern: &str, is_active: bool) -> SpicyResult<Series> {
        let mut jobs = self.job.write().unwrap();
        let mut ids = vec![];
        for (id, job) in jobs.iter_mut() {
            if job.description.contains(pattern) {
                job.is_active = is_active;
                ids.push(*id);
            }
        }
        Ok(Series::from_vec("id".into(), ids))
    }

    pub fn clear_job(&self) -> SpicyResult<()> {
        let mut jobs = self.job.write().unwrap();
        jobs.clear();
        Ok(())
    }

    pub fn validate_auth_token(&self, stream: &mut TcpStream, users: &[String]) -> AuthInfo {
        let mut default_auth = AuthInfo {
            username: String::from("anonymous"),
            is_authenticated: false,
            version: 0,
        };

        let mut buffer = [0; 1024];

        match stream.read(&mut buffer) {
            Ok(n) => {
                let credentials = String::from_utf8_lossy(&buffer[..n - 2]).trim().to_string();
                let version = buffer[n - 2];
                if version < 3 {
                    error!(
                        "{} with version {} is not supported",
                        stream.peer_addr().unwrap(),
                        version
                    );
                    default_auth.version = version;
                    return default_auth;
                } else {
                    info!(
                        "{} with version {} connected",
                        stream.peer_addr().unwrap(),
                        version
                    );
                }

                let parts: Vec<&str> = credentials.splitn(2, ':').collect();

                let (username, _) = if parts.len() == 1 {
                    (parts[0], "")
                } else {
                    (parts[0], parts[1])
                };

                info!("validating auth token for user '{}'", username);

                if users.is_empty() || users.contains(&username.to_string()) {
                    return AuthInfo {
                        username: username.to_string(),
                        is_authenticated: true,
                        version,
                    };
                }
                default_auth
            }
            Err(e) => {
                error!("error reading auth credentials: {}", e);
                stream.shutdown(std::net::Shutdown::Both).unwrap();
                default_auth
            }
        }
    }

    pub fn stats(&self) -> SpicyResult<SpicyObj> {
        let mut status = IndexMap::new();
        status.insert("lazy_mode".into(), SpicyObj::Boolean(self.is_lazy_mode()));
        status.insert(
            "repl_lang".into(),
            SpicyObj::String(self.repl_lang.as_str().to_owned()),
        );
        status.insert(
            "partitioned_df_count".into(),
            SpicyObj::I64(self.par_df.read().unwrap().len() as i64),
        );
        status.insert(
            "parse_cache_len".into(),
            SpicyObj::I64(self.parse_cache.lock().unwrap().len() as i64),
        );
        status.insert(
            "partitioned_df_paths".into(),
            SpicyObj::Series(Series::new(
                "path".into(),
                self.par_df
                    .read()
                    .unwrap()
                    .iter()
                    .map(|(_, p)| p.path.clone())
                    .collect::<Vec<String>>(),
            )),
        );
        Ok(SpicyObj::Dict(status))
    }

    /// Start a TCP listener on the given port and spawn a thread per
    /// incoming connection. Handles authentication, IPC version
    /// negotiation, and connection dispatch.
    ///
    /// This method blocks the **calling thread** (it should be spawned
    /// on a dedicated thread from `main`).
    pub fn start_tcp_listener(self: &Arc<Self>, port: i32, remote: bool, users: Vec<String>) {
        let addr = if remote {
            format!("0.0.0.0:{}", port)
        } else {
            format!("127.0.0.1:{}", port)
        };

        let listener = match TcpListener::bind(&addr) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("{} - {}", e, port);
                std::process::exit(1)
            }
        };

        info!("listening at port {}", port);

        for stream in listener.incoming() {
            let state_tcp = Arc::clone(self);
            let mut stream = stream.unwrap();
            let auth_info = state_tcp.validate_auth_token(&mut stream, &users);
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
            let ipc_type = IpcType::from_u8(auth_info.version)
                .unwrap_or_else(|| panic!("unsupported ipc version: {}", auth_info.version));
            let h = state_tcp
                .set_handle(
                    Some(Box::new(stream.try_clone().unwrap())),
                    &peer_addr,
                    &format!("{}://{}", ipc_type, peer_addr,),
                    false,
                    ipc_type,
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
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpcType {
    // compatible with kdb+
    Q = 0,
    Chili = 1,
}

impl IpcType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            3 => Some(IpcType::Q),
            6 => Some(IpcType::Q),
            9 => Some(IpcType::Chili),
            _ => None,
        }
    }
}

impl Display for IpcType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpcType::Q => write!(f, "q"),
            IpcType::Chili => write!(f, "chili"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnType {
    Incoming = 0,
    Outgoing = 1,
    Publishing = 2,
    Subscribing = 3,
    Disconnected = 4,
    File = 5,
    Sequence = 6,
    New = 7,
}
