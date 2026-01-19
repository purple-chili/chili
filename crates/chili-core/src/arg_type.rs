use std::fmt::Display;

use crate::{SpicyError, SpicyResult, obj::SpicyObj};

pub enum ArgType {
    Any,
    Boolean,
    Int,
    Expr,
    Float,
    IntLike,
    NumericLike,
    NumericNative,
    NumericNativeSeries,
    Dict,
    DictOrSeries,
    DataFrame,
    LazyFrame,
    DataFrameOrSeries,
    DataFrameOrMatrix,
    DataFrameOrList,
    Series,
    Str,
    StrLike,
    StrOrStrs,
    StrOrSym,
    Sym,
    SymOrSyms,
    Syms,
    Timestamp,
    TimestampLike,
    Duration,
}

impl Display for ArgType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ArgType::Any => "args",
            ArgType::Boolean => "bool",
            ArgType::Int => "bool | int",
            ArgType::Expr => "expr",
            ArgType::Float => "bool | int | float",
            ArgType::IntLike => "bool(s) | int(s)",
            ArgType::NumericLike => "bool(s) | int(s) | float(s)",
            ArgType::NumericNative => "bool(s) | int(s) | float(s) | temporal(s)",
            ArgType::NumericNativeSeries => "ints | floats | temporals",
            ArgType::Dict => "dict",
            ArgType::DictOrSeries => "dict | series",
            ArgType::DataFrame => "dataframe",
            ArgType::LazyFrame => "lazyframe",
            ArgType::DataFrameOrSeries => "dataframe | series",
            ArgType::DataFrameOrMatrix => "dataframe | matrix",
            ArgType::DataFrameOrList => "dataframe | list",
            ArgType::Series => "series",
            ArgType::Str => "str",
            ArgType::StrLike => "str(s) | sym(s)",
            ArgType::StrOrStrs => "str | str(s)",
            ArgType::StrOrSym => "str | sym",
            ArgType::Sym => "sym",
            ArgType::SymOrSyms => "sym(s)",
            ArgType::Syms => "syms",
            ArgType::Timestamp => "timestamp",
            ArgType::TimestampLike => "datetime(s) | timestamp(s)",
            ArgType::Duration => "duration",
        };
        write!(f, "{}", s)
    }
}

impl ArgType {
    pub fn validate(&self, args: &SpicyObj) -> bool {
        match self {
            ArgType::Any => true,
            ArgType::Boolean => args.is_bool(),
            ArgType::Int => args.is_integer() || args.is_bool(),
            ArgType::IntLike => {
                let code = args.get_type_code();
                (-5..=5).contains(&code)
            }
            ArgType::Expr => args.is_expr(),
            ArgType::Float => args.is_numeric() || args.is_bool(),
            ArgType::NumericLike => args.is_numeric_like(),
            ArgType::NumericNative => {
                let code = args.get_type_code();
                (-12..=12).contains(&code)
            }
            ArgType::NumericNativeSeries => {
                let code = args.get_type_code();
                (2..=18).contains(&code) && code != 13 && code != 14
            }
            ArgType::Dict => args.is_dict(),
            ArgType::DictOrSeries => args.is_dict() || args.is_series(),
            ArgType::DataFrame => args.is_df(),
            ArgType::LazyFrame => args.is_lf(),
            ArgType::DataFrameOrSeries => args.is_df() || args.is_series(),
            ArgType::DataFrameOrMatrix => args.is_df() || args.is_matrix(),
            ArgType::DataFrameOrList => args.is_df() || args.is_mixed_list(),
            ArgType::Series => args.is_series(),
            ArgType::Str => args.is_str(),
            ArgType::StrLike => args.is_str_like(),
            ArgType::StrOrStrs => args.is_str_or_strs(),
            ArgType::StrOrSym => args.is_str() | args.is_sym(),
            ArgType::Sym => args.is_sym(),
            ArgType::SymOrSyms => args.is_sym_or_syms(),
            ArgType::Syms => args.is_syms() || (args.is_mixed_list() && args.size() == 0),
            ArgType::Timestamp => args.is_timestamp(),
            ArgType::TimestampLike => args.is_timestamp() || args.is_datetime(),
            ArgType::Duration => args.is_duration(),
        }
    }
}

pub fn validate_args(args: &[&SpicyObj], cfg: &[ArgType]) -> SpicyResult<()> {
    let err_msg = args
        .iter()
        .enumerate()
        .zip(cfg)
        .map(|((i, any), arg_type)| raise_arg_type_err(any, i, arg_type))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if err_msg.is_empty() {
        Ok(())
    } else {
        Err(SpicyError::Err(err_msg))
    }
}

pub fn raise_arg_type_err(args: &SpicyObj, arg_pos: usize, arg_type: &ArgType) -> String {
    if arg_type.validate(args) {
        "".to_owned()
    } else {
        format!(
            "Expect data type '{0}' for '{1}' argument , got '{2}'.",
            arg_type,
            arg_pos + 1,
            args.get_type_name()
        )
    }
}
