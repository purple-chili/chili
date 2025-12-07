use std::collections::HashMap;
use std::sync::LazyLock;

use chili_core::{Func, SpicyObj, SpicyResult};

pub fn log_str(args: &[&SpicyObj]) -> SpicyResult<String> {
    let arg0 = args[0];
    match arg0 {
        SpicyObj::String(s) => Ok(s.to_owned()),
        SpicyObj::MixedList(v) => {
            let s = v
                .iter()
                .map(|args| {
                    if let SpicyObj::String(s) = args {
                        s.as_str().to_owned()
                    } else {
                        args.to_string()
                    }
                })
                .collect::<Vec<String>>()
                .join(" ");
            Ok(s)
        }
        _ => Ok(arg0.to_string()),
    }
}
pub fn debug(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let s = log_str(args)?;
    log::debug!("{}", s);
    Ok(SpicyObj::Null)
}

pub fn info(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let s = log_str(args)?;
    log::info!("{}", s);
    Ok(SpicyObj::Null)
}

pub fn warn(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let s = log_str(args)?;
    log::warn!("{}", s);
    Ok(SpicyObj::Null)
}

pub fn error(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let s = log_str(args)?;
    log::error!("{}", s);
    Ok(SpicyObj::Null)
}

pub static LOG_FN: LazyLock<HashMap<String, Func>> = LazyLock::new(|| {
    [
        (
            ".log.info".to_owned(),
            Func::new_built_in_fn(Some(Box::new(info)), 1, ".log.info", &["str"]),
        ),
        (
            ".log.warn".to_owned(),
            Func::new_built_in_fn(Some(Box::new(warn)), 1, ".log.warn", &["str"]),
        ),
        (
            ".log.debug".to_owned(),
            Func::new_built_in_fn(Some(Box::new(debug)), 1, ".log.debug", &["str"]),
        ),
        (
            ".log.error".to_owned(),
            Func::new_built_in_fn(Some(Box::new(error)), 1, ".log.error", &["str"]),
        ),
    ]
    .into_iter()
    .collect()
});
