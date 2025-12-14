use std::num::ParseIntError;

use pest::Span;
use pest::error::{Error as PestError, ErrorVariant};
use pest::{Parser, iterators::Pair};
use polars::datatypes::{
    DataType as PolarsDataType, Float64Type, Int16Type, Int32Type, Int64Type, TimeUnit,
};
use polars::frame::DataFrame;
use polars::prelude::col;
use polars::prelude::{
    Categories, Column, IndexOrder, Int8Type, Int128Type, NamedFrom, NamedFromOwned, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use polars::series::Series;

use crate::SpicyError;
use crate::ast_node::{AstNode, QueryOp, SourcePos};
use crate::func::Func;
use crate::obj::SpicyObj;

#[cfg(not(feature = "vintage"))]
use chili_grammar::{ChiliParser, Rule};
#[cfg(feature = "vintage")]
use chili_vintage_grammar::{ChiliParser, Rule};

fn parse_binary_exp(pair: Pair<Rule>, source_id: usize) -> Result<AstNode, SpicyError> {
    match pair.as_rule() {
        Rule::BinaryOp | Rule::BinaryKeyword => Ok(AstNode::Id {
            pos: SourcePos::new(pair.as_span().start(), source_id),
            name: pair.as_str().to_owned(),
        }),
        Rule::Fn => parse_exp(pair, source_id),
        _ => Err(raise_error(
            format!("Unexpected binary op/function: {}", pair.as_str()),
            pair.as_span(),
        )),
    }
}

fn parse_exp(pair: Pair<Rule>, source_id: usize) -> Result<AstNode, SpicyError> {
    let rule = pair.as_rule();
    match rule {
        Rule::Exp => parse_exp(pair.into_inner().next().unwrap(), source_id),
        #[cfg(feature = "vintage")]
        Rule::UnaryExp | Rule::UnaryQueryExp => {
            let mut pair = pair.into_inner();
            let unary = pair.next().unwrap();
            let exp = pair.next().unwrap();
            let exp = parse_exp(exp, source_id)?;
            Ok(AstNode::UnaryExp {
                f: Box::new(parse_exp(unary, source_id)?),
                exp: Box::new(exp),
            })
        }
        #[cfg(feature = "vintage")]
        Rule::BinaryExp | Rule::BinaryQueryExp => {
            let mut pair = pair.into_inner();
            let lhs_pair = pair.next().unwrap();
            let lhs = parse_exp(lhs_pair, source_id)?;
            let binary_exp = pair.next().unwrap();
            let rhs_pair = pair.next().unwrap();
            let rhs = parse_exp(rhs_pair, source_id)?;
            if ["||", "&&", "??"].contains(&binary_exp.as_str()) {
                Ok(AstNode::ShortCircuit {
                    op: binary_exp.as_str().to_owned(),
                    left_cond: Box::new(lhs),
                    right_cond: Box::new(rhs),
                })
            } else {
                Ok(AstNode::BinaryExp {
                    f2: Box::new(parse_binary_exp(binary_exp, source_id)?),
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }
        }
        // left associative binary expressions
        #[cfg(not(feature = "vintage"))]
        Rule::BinaryExp | Rule::BinaryQueryExp => {
            let mut pair = pair.into_inner();
            let mut prev_node: AstNode;
            let first_lhs_pair = pair.next().unwrap();
            prev_node = parse_exp(first_lhs_pair, source_id)?;

            for _ in 0..(pair.len() / 2) {
                let binary_exp = pair.next().unwrap();

                let rhs_pair = pair.next().unwrap();
                let rhs = parse_exp(rhs_pair, source_id)?;
                if ["||", "&&", "??"].contains(&binary_exp.as_str()) {
                    prev_node = AstNode::ShortCircuit {
                        op: binary_exp.as_str().to_owned(),
                        left_cond: Box::new(prev_node),
                        right_cond: Box::new(rhs),
                    };
                } else {
                    prev_node = AstNode::BinaryExp {
                        f2: Box::new(parse_binary_exp(binary_exp, source_id)?),
                        lhs: Box::new(prev_node),
                        rhs: Box::new(rhs),
                    };
                }
            }
            Ok(prev_node)
        }
        #[cfg(feature = "vintage")]
        Rule::EmptyList => parse_spicy_obj(pair),
        Rule::Boolean
        | Rule::U8
        | Rule::I16
        | Rule::I32
        | Rule::I64
        | Rule::F32
        | Rule::F64
        | Rule::Date
        | Rule::Time
        | Rule::Datetime
        | Rule::Timestamp
        | Rule::Duration
        | Rule::Booleans
        | Rule::U8s
        | Rule::Integers
        | Rule::Floats
        | Rule::Dates
        | Rule::Times
        | Rule::Datetimes
        | Rule::Timestamps
        | Rule::Durations
        | Rule::Sym
        | Rule::Syms
        | Rule::String => parse_spicy_obj(pair),
        Rule::AssignmentExp => {
            let mut pairs = pair.into_inner();
            let id = pairs.next().unwrap();
            if id.as_rule() == Rule::FnCall {
                let mut fn_call = id.into_inner();
                let id = fn_call.next().unwrap().as_str();
                let mut indices: Vec<AstNode> = Vec::with_capacity(fn_call.len() - 1);
                for arg in fn_call {
                    indices.push(parse_exp(arg.into_inner().next().unwrap(), source_id)?)
                }
                let exp = parse_exp(pairs.next().unwrap(), source_id)?;
                Ok(AstNode::IndexAssignmentExp {
                    id: id.to_owned(),
                    indices,
                    exp: Box::new(exp),
                })
            } else {
                let exp = pairs.next().unwrap();
                let exp = parse_exp(exp, source_id)?;
                Ok(AstNode::AssignmentExp {
                    id: id.as_str().to_owned(),
                    exp: Box::new(exp),
                })
            }
        }
        Rule::Column => {
            let column = pair.as_str().to_owned();
            Ok(AstNode::SpicyObj(SpicyObj::Expr(col(column
                [1..column.len() - 1]
                .to_owned()))))
        }
        Rule::Id => {
            let id = pair.as_str().to_owned();
            Ok(AstNode::Id {
                name: id,
                pos: SourcePos::new(pair.as_span().start(), source_id),
            })
        }
        Rule::Fn => {
            let fn_body = pair.as_str();
            let fn_span = pair.as_span();
            let mut pairs = pair.into_inner();
            let pair = pairs.next().unwrap();
            let inner = pair.into_inner();
            let mut params: Vec<String> = Vec::with_capacity(inner.len());
            for pair in inner {
                params.push(pair.as_str().to_owned())
            }
            let mut nodes = Vec::with_capacity(pairs.len() - 1);
            for pair in pairs {
                nodes.push(parse_exp(pair, source_id)?)
            }
            Ok(AstNode::SpicyObj(SpicyObj::Fn(Func::new(
                fn_body,
                params,
                nodes,
                SourcePos::new(fn_span.start(), source_id),
            ))))
        }
        Rule::FnCall => {
            let span = pair.as_span().start();
            let mut pairs = pair.into_inner();
            let f = parse_exp(pairs.next().unwrap(), source_id)?;
            let mut args = Vec::with_capacity(pairs.len() - 1);
            for pair in pairs {
                args.push(parse_exp(pair.into_inner().next().unwrap(), source_id)?)
            }
            // if f is eval, and first args is J::String, parse J::string
            Ok(AstNode::FnCall {
                pos: SourcePos::new(span, source_id),
                f: Box::new(f),
                args,
            })
        }
        Rule::IfExp => {
            let mut pairs = pair.into_inner();
            if pairs.len() == 1 {
                let mut pairs = pairs.next().unwrap().into_inner();
                let cond = parse_exp(pairs.next().unwrap(), source_id)?;
                let mut nodes = Vec::new();
                for pair in pairs {
                    let rule = pair.as_rule();
                    nodes.push(parse_exp(pair, source_id)?);
                    if rule == Rule::ReturnExp {
                        break;
                    }
                }
                Ok(AstNode::If {
                    cond: Box::new(cond),
                    nodes,
                    else_nodes: vec![],
                })
            } else {
                let cond = parse_exp(pairs.next().unwrap(), source_id)?;
                let mut nodes = Vec::new();
                for pair in pairs.next().unwrap().into_inner() {
                    let rule = pair.as_rule();
                    nodes.push(parse_exp(pair, source_id)?);
                    if rule == Rule::ReturnExp {
                        break;
                    }
                }
                let mut else_nodes = Vec::new();
                if let Some(pair) = pairs.next() {
                    for pair in pair.into_inner() {
                        let rule = pair.as_rule();
                        else_nodes.push(parse_exp(pair, source_id)?);
                        if rule == Rule::ReturnExp {
                            break;
                        }
                    }
                }
                Ok(AstNode::If {
                    cond: Box::new(cond),
                    nodes,
                    else_nodes,
                })
            }
        }
        Rule::WhileExp => {
            let mut pairs = pair.into_inner();
            let cond = parse_exp(pairs.next().unwrap(), source_id)?;
            let mut nodes = Vec::new();
            for pair in pairs.next().unwrap().into_inner() {
                let rule = pair.as_rule();
                nodes.push(parse_exp(pair, source_id)?);
                if rule == Rule::ReturnExp {
                    break;
                }
            }
            Ok(AstNode::While {
                cond: Box::new(cond),
                nodes,
            })
        }
        Rule::IfElseExp => {
            let pairs = pair.into_inner();
            let mut nodes = Vec::new();
            for pair in pairs {
                nodes.push(parse_exp(pair, source_id)?);
            }
            Ok(AstNode::IfElse { nodes })
        }
        Rule::TryExp => {
            let mut pairs = pair.into_inner();
            let length = pairs.len();
            let mut tries = Vec::new();
            let mut catches = Vec::new();
            for pair in pairs.next().unwrap().into_inner() {
                tries.push(parse_exp(pair, source_id)?);
            }
            let err_id = if length == 3 {
                pairs.next().unwrap().as_str().to_owned()
            } else {
                "err".to_owned()
            };
            for pair in pairs.next().unwrap().into_inner() {
                catches.push(parse_exp(pair, source_id)?);
            }
            Ok(AstNode::Try {
                tries,
                err_id,
                catches,
            })
        }
        Rule::ReturnExp => {
            let node = parse_exp(pair.into_inner().next().unwrap(), source_id)?;
            Ok(AstNode::Return(Box::new(node)))
        }
        Rule::RaiseExp => {
            let node = parse_exp(pair.into_inner().next().unwrap(), source_id)?;
            Ok(AstNode::Raise(Box::new(node)))
        }
        Rule::Null => Ok(AstNode::SpicyObj(SpicyObj::Null)),
        Rule::DelayedArg => Ok(AstNode::DelayedArg),
        Rule::Table => {
            let span = pair.as_span();
            let cols = pair.into_inner();
            let mut col_exps: Vec<AstNode> = Vec::with_capacity(cols.len());
            let mut all_series = true;
            for (i, col_exp) in cols.enumerate() {
                let name: String;
                let exp: AstNode;
                let node = col_exp.into_inner().next().unwrap();
                let is_rename = node.as_rule() == Rule::RenameColExp;
                if is_rename {
                    let mut nodes = node.into_inner();
                    name = nodes.next().unwrap().as_str().to_owned();
                    exp = parse_exp(nodes.next().unwrap(), source_id)?;
                } else {
                    name = format!("col{:02}", i);
                    exp = parse_exp(node, source_id)?
                }
                if let AstNode::SpicyObj(obj) = exp {
                    if let SpicyObj::Series(mut s) = obj {
                        s.rename(name.into());
                        col_exps.push(AstNode::SpicyObj(SpicyObj::Series(s)));
                    } else {
                        let mut s = obj
                            .into_series()
                            .map_err(|e| raise_error(e.to_string(), span))?;
                        s.rename(name.into());
                        col_exps.push(AstNode::SpicyObj(SpicyObj::Series(s)));
                    }
                } else if let AstNode::Id { name: id, .. } = &exp {
                    let name = if is_rename { name } else { id.to_owned() };
                    col_exps.push(AstNode::ColExp {
                        name,
                        exp: Box::new(exp),
                    });
                    all_series = false;
                } else {
                    col_exps.push(AstNode::ColExp {
                        name,
                        exp: Box::new(exp),
                    });
                    all_series = false;
                }
            }
            if all_series {
                let series: Vec<Column> = col_exps
                    .into_iter()
                    .map(|node| node.spicy_obj().unwrap().series().unwrap().clone().into())
                    .collect();
                let df = match DataFrame::new(series) {
                    Ok(df) => df,
                    Err(e) => return Err(raise_error(e.to_string(), span)),
                };
                Ok(AstNode::SpicyObj(SpicyObj::DataFrame(df)))
            } else {
                Ok(AstNode::Table(col_exps))
            }
        }
        Rule::Matrix => {
            let span = pair.as_span();
            let cols = pair.into_inner();
            let mut exps: Vec<AstNode> = Vec::with_capacity(cols.len());
            let mut all_series = true;
            for (i, col_exp) in cols.enumerate() {
                let node = col_exp.into_inner().next().unwrap();
                let node_span = node.as_span();
                let col_name: String = format!("col{:02}", i);
                let exp: AstNode = parse_exp(node, source_id)?;
                if let AstNode::SpicyObj(obj) = exp {
                    let type_name = obj.get_type_name();
                    if let SpicyObj::Series(mut s) = obj {
                        if !(s.dtype().is_primitive_numeric() || s.dtype().is_bool()) {
                            return Err(raise_error(
                                format!("Requires numeric data type, got '{}'", type_name),
                                node_span,
                            ));
                        }
                        s.rename(col_name.into());
                        exps.push(AstNode::SpicyObj(SpicyObj::Series(s)));
                    } else {
                        if !(obj.is_numeric() || obj.is_bool()) {
                            return Err(raise_error(
                                format!("Requires numeric data type, got '{}'", type_name),
                                node_span,
                            ));
                        }
                        let mut s = obj.into_series().unwrap();
                        s.rename(col_name.into());
                        exps.push(AstNode::SpicyObj(SpicyObj::Series(s)));
                    }
                } else {
                    exps.push(AstNode::ColExp {
                        name: col_name,
                        exp: Box::new(exp),
                    });
                    all_series = false;
                }
            }
            if all_series {
                let cols: Vec<Column> = exps
                    .into_iter()
                    .map(|node| node.spicy_obj().unwrap().series().unwrap().clone().into())
                    .collect();
                let df = match DataFrame::new(cols) {
                    Ok(df) => df,
                    Err(e) => return Err(raise_error(e.to_string(), span)),
                };
                let matrix = df
                    .to_ndarray::<Float64Type>(IndexOrder::C)
                    .map_err(|e| raise_error(e.to_string(), span))?;
                Ok(AstNode::SpicyObj(SpicyObj::Matrix(
                    matrix.reversed_axes().to_shared(),
                )))
            } else {
                Ok(AstNode::Matrix(exps))
            }
        }
        Rule::Query => parse_query(pair, source_id),
        Rule::BinaryKeyword => Ok(AstNode::Id {
            pos: SourcePos::new(pair.as_span().start(), source_id),
            name: pair.as_str().to_owned(),
        }),
        Rule::BracketExp => Ok(parse_exp(pair.into_inner().next().unwrap(), source_id)?),
        Rule::ListExp => {
            let pairs = pair.into_inner();
            let mut list = Vec::with_capacity(pairs.len());
            for pair in pairs {
                list.push(parse_list(pair, source_id)?)
            }
            Ok(AstNode::List(list))
        }
        Rule::Dict => {
            let pairs = pair.into_inner();
            let mut keys: Vec<String> = Vec::with_capacity(pairs.len());
            let mut values: Vec<AstNode> = Vec::with_capacity(pairs.len());
            for pair in pairs {
                let mut kv = pair.into_inner();
                keys.push(kv.next().unwrap().as_str().to_owned());
                values.push(parse_exp(kv.next().unwrap(), source_id)?)
            }
            Ok(AstNode::Dict { keys, values })
        }
        Rule::NullExp => Ok(AstNode::SpicyObj(SpicyObj::Null)),
        Rule::GlobalId => Ok(AstNode::Id {
            name: pair.as_str().to_owned(),
            pos: SourcePos::new(pair.as_span().start(), source_id),
        }),
        unexpected_exp => Err(raise_error(
            format!("Unexpected rule: {:?}", unexpected_exp),
            pair.as_span(),
        )),
    }
}

fn parse_list(pair: Pair<Rule>, source_id: usize) -> Result<AstNode, SpicyError> {
    match pair.as_rule() {
        Rule::BinaryOp | Rule::BinaryKeyword => Ok(AstNode::Id {
            name: pair.as_str().to_owned(),
            pos: SourcePos::new(pair.as_span().start(), source_id),
        }),
        Rule::Exp => parse_exp(pair, source_id),
        _ => Err(raise_error(
            format!("Unexpected list expression: {:?}", pair.as_str()),
            pair.as_span(),
        )),
    }
}

fn parse_spicy_obj(pair: Pair<Rule>) -> Result<AstNode, SpicyError> {
    let s = pair.as_str();
    match pair.as_rule() {
        #[cfg(feature = "vintage")]
        Rule::EmptyList => Ok(AstNode::SpicyObj(SpicyObj::MixedList(vec![]))),
        Rule::Boolean => Ok(AstNode::SpicyObj(SpicyObj::Boolean(pair.as_str() == "1b"))),
        Rule::U8 => match u8::from_str_radix(&pair.as_str()[2..], 16) {
            Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::U8(n))),
            Err(e) => Err(raise_error(e.to_string(), pair.as_span())),
        },
        Rule::I16 => {
            let s = if let Some(num) = s.strip_suffix("i16") {
                num
            } else {
                &s[..s.len() - 1]
            };
            match s.parse::<i16>() {
                Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::I16(n))),
                Err(e) => Err(raise_error(e.to_string(), pair.as_span())),
            }
        }
        Rule::I32 => {
            let s = if let Some(num) = s.strip_suffix("i32") {
                num
            } else {
                &s[..s.len() - 1]
            };
            match s.parse::<i32>() {
                Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::I32(n))),
                Err(e) => Err(raise_error(e.to_string(), pair.as_span())),
            }
        }
        Rule::I64 => match pair.as_str().parse::<i64>() {
            Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::I64(n))),
            Err(e) => Err(raise_error(e.to_string(), pair.as_span())),
        },
        Rule::F32 => {
            let s = if let Some(num) = s.strip_suffix("f32") {
                num
            } else {
                &s[..s.len() - 1]
            };
            if s == "-0w" {
                Ok(AstNode::SpicyObj(SpicyObj::F32(f32::NEG_INFINITY)))
            } else if s == "0w" {
                Ok(AstNode::SpicyObj(SpicyObj::F32(f32::INFINITY)))
            } else {
                match s.parse::<f32>() {
                    Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::F32(n))),
                    Err(e) => Err(raise_error(e.to_string(), pair.as_span())),
                }
            }
        }
        Rule::F64 => {
            if s == "-0w" {
                Ok(AstNode::SpicyObj(SpicyObj::F64(f64::NEG_INFINITY)))
            } else if s == "0w" {
                Ok(AstNode::SpicyObj(SpicyObj::F64(f64::INFINITY)))
            } else {
                match pair.as_str().parse::<f64>() {
                    Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::F64(n))),
                    Err(e) => Err(raise_error(e.to_string(), pair.as_span())),
                }
            }
        }
        Rule::Date => {
            let obj = SpicyObj::parse_date(pair.as_str())
                .map_err(|e| raise_error(e.to_string(), pair.as_span()))?;
            Ok(AstNode::SpicyObj(obj))
        }
        Rule::Time => {
            let obj = SpicyObj::parse_time(pair.as_str())
                .map_err(|e| raise_error(e.to_string(), pair.as_span()))?;
            Ok(AstNode::SpicyObj(obj))
        }
        Rule::Datetime => {
            let obj = SpicyObj::parse_datetime(pair.as_str())
                .map_err(|e| raise_error(e.to_string(), pair.as_span()))?;
            Ok(AstNode::SpicyObj(obj))
        }
        Rule::Timestamp => {
            let obj = SpicyObj::parse_timestamp(pair.as_str())
                .map_err(|e| raise_error(e.to_string(), pair.as_span()))?;
            Ok(AstNode::SpicyObj(obj))
        }
        Rule::Duration => {
            let obj = SpicyObj::parse_duration(pair.as_str())
                .map_err(|e| raise_error(e.to_string(), pair.as_span()))?;
            Ok(AstNode::SpicyObj(obj))
        }
        Rule::Dates => {
            let dates = pair
                .as_str()
                .split_whitespace()
                .map(|s| {
                    SpicyObj::parse_date(s).map_err(|e| raise_error(e.to_string(), pair.as_span()))
                })
                .collect::<Result<Vec<SpicyObj>, _>>()?
                .iter()
                .map(|args| args.to_i64().unwrap() as i32)
                .collect::<Vec<i32>>();
            Ok(AstNode::SpicyObj(SpicyObj::Series(
                Series::new("".into(), dates)
                    .cast(&PolarsDataType::Date)
                    .map_err(|e| raise_error(e.to_string(), pair.as_span()))?,
            )))
        }
        Rule::Times => {
            let times = pair
                .as_str()
                .split_whitespace()
                .map(|s| {
                    SpicyObj::parse_time(s).map_err(|e| raise_error(e.to_string(), pair.as_span()))
                })
                .collect::<Result<Vec<SpicyObj>, _>>()?
                .iter()
                .map(|args| args.to_i64().unwrap())
                .collect::<Vec<i64>>();
            Ok(AstNode::SpicyObj(SpicyObj::Series(
                Series::new("".into(), times)
                    .cast(&PolarsDataType::Time)
                    .map_err(|e| raise_error(e.to_string(), pair.as_span()))?,
            )))
        }
        Rule::Durations => {
            let times = pair
                .as_str()
                .split_whitespace()
                .map(|s| {
                    SpicyObj::parse_duration(s)
                        .map_err(|e| raise_error(e.to_string(), pair.as_span()))
                })
                .collect::<Result<Vec<SpicyObj>, _>>()?
                .iter()
                .map(|args| args.to_i64().unwrap())
                .collect::<Vec<i64>>();
            Ok(AstNode::SpicyObj(SpicyObj::Series(
                Series::new("".into(), times)
                    .cast(&PolarsDataType::Duration(TimeUnit::Nanoseconds))
                    .map_err(|e| raise_error(e.to_string(), pair.as_span()))?,
            )))
        }
        Rule::Datetimes => {
            let datetimes = pair
                .as_str()
                .split_whitespace()
                .map(|s| {
                    SpicyObj::parse_datetime(s)
                        .map_err(|e| raise_error(e.to_string(), pair.as_span()))
                })
                .collect::<Result<Vec<SpicyObj>, _>>()?
                .iter()
                .map(|args| args.to_i64().unwrap())
                .collect::<Vec<i64>>();
            Ok(AstNode::SpicyObj(SpicyObj::Series(
                Series::new("".into(), datetimes)
                    .cast(&PolarsDataType::Datetime(TimeUnit::Milliseconds, None))
                    .map_err(|e| raise_error(e.to_string(), pair.as_span()))?,
            )))
        }
        Rule::Timestamps => {
            let timestamps = pair
                .as_str()
                .split_whitespace()
                .map(|s| {
                    SpicyObj::parse_timestamp(s)
                        .map_err(|e| raise_error(e.to_string(), pair.as_span()))
                })
                .collect::<Result<Vec<SpicyObj>, _>>()?
                .iter()
                .map(|args| args.to_i64().unwrap())
                .collect::<Vec<i64>>();
            Ok(AstNode::SpicyObj(SpicyObj::Series(
                Series::new("".into(), timestamps)
                    .cast(&PolarsDataType::Datetime(TimeUnit::Nanoseconds, None))
                    .map_err(|e| raise_error(e.to_string(), pair.as_span()))?,
            )))
        }
        Rule::Booleans => {
            let s = pair.as_str();
            let b: Vec<bool> = s
                .as_bytes()
                .iter()
                .take(s.len() - 1)
                .map(|u| *u == b'1')
                .collect();
            let s = Series::new("".into(), b);
            Ok(AstNode::SpicyObj(SpicyObj::Series(s)))
        }
        Rule::U8s => {
            let s = &pair.as_str()[2..];
            match (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
                .collect::<Result<Vec<u8>, ParseIntError>>()
            {
                Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::Series(Series::from_vec(
                    "".into(),
                    n,
                )))),
                Err(e) => Err(raise_error(e.to_string(), pair.as_span())),
            }
        }
        Rule::Integers => {
            let obj = if let Some(numbers) = s.strip_suffix("h") {
                SpicyObj::parse_numeric_series::<i16, Int16Type>(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("i") {
                SpicyObj::parse_numeric_series::<i32, Int32Type>(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("u8") {
                SpicyObj::parse_numeric_series::<u8, UInt8Type>(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("u16") {
                SpicyObj::parse_numeric_series::<u16, UInt16Type>(numbers)
                    .map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("u32") {
                SpicyObj::parse_numeric_series::<u32, UInt32Type>(numbers)
                    .map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("u64") {
                SpicyObj::parse_numeric_series::<u64, UInt64Type>(numbers)
                    .map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("i8") {
                SpicyObj::parse_numeric_series::<i8, Int8Type>(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("i16") {
                SpicyObj::parse_numeric_series::<i16, Int16Type>(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("i32") {
                SpicyObj::parse_numeric_series::<i32, Int32Type>(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("i64") {
                SpicyObj::parse_numeric_series::<i64, Int64Type>(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("i128") {
                SpicyObj::parse_numeric_series::<i128, Int128Type>(numbers)
                    .map_err(|e| e.to_string())
            } else {
                SpicyObj::parse_numeric_series::<i64, Int64Type>(s).map_err(|e| e.to_string())
            };
            match obj {
                Ok(obj) => Ok(AstNode::SpicyObj(obj)),
                Err(e) => Err(raise_error(e, pair.as_span())),
            }
        }
        Rule::Floats => {
            let obj = if let Some(numbers) = s.strip_suffix("e") {
                SpicyObj::parse_numeric_series_f32(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("f") {
                SpicyObj::parse_numeric_series_f64(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("f32") {
                SpicyObj::parse_numeric_series_f32(numbers).map_err(|e| e.to_string())
            } else if let Some(numbers) = s.strip_suffix("f64") {
                SpicyObj::parse_numeric_series_f64(numbers).map_err(|e| e.to_string())
            } else {
                SpicyObj::parse_numeric_series_f64(s).map_err(|e| e.to_string())
            };
            match obj {
                Ok(obj) => Ok(AstNode::SpicyObj(obj)),
                Err(e) => Err(raise_error(e, pair.as_span())),
            }
        }
        Rule::Sym => Ok(AstNode::SpicyObj(SpicyObj::Symbol(
            pair.as_str()[1..].to_string(),
        ))),
        Rule::Syms => {
            let syms = pair.as_str()[1..].split("`").collect::<Vec<_>>();
            Ok(AstNode::SpicyObj(SpicyObj::Series(
                Series::new("".into(), syms)
                    .cast(&PolarsDataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
            )))
        }
        Rule::String => {
            let str = pair.as_str();
            // Strip leading and ending quotes.
            let str = &str[1..str.len() - 1];
            // Escaped string quotes become single quotes here.
            Ok(AstNode::SpicyObj(SpicyObj::String(str.to_owned())))
        }
        unexpected_rule => Err(raise_error(
            format!("Unexpected rule for spicy obj: {:?}", unexpected_rule),
            pair.as_span(),
        )),
    }
}

fn parse_query(pair: Pair<Rule>, source_id: usize) -> Result<AstNode, SpicyError> {
    let mut pairs = pair.into_inner();
    // select, update, exec, delete
    let op = QueryOp::from_str(pairs.next().unwrap().as_str()).unwrap();
    let mut op_exp: Vec<AstNode> = Vec::new();
    let mut by_exp: Vec<AstNode> = Vec::new();
    let mut from_exp: AstNode = AstNode::DelayedArg;
    let mut where_exp: Vec<AstNode> = Vec::new();
    let mut limited_exp: Option<AstNode> = None;
    for some_pair in pairs {
        match some_pair.as_rule() {
            Rule::SelectExp | Rule::ColNames => {
                let op_pairs = some_pair.into_inner();
                for op_pair in op_pairs {
                    op_exp.push(parse_query_col_exp(op_pair, source_id)?)
                }
            }
            Rule::ByExp => {
                let by_pairs = some_pair.into_inner();
                by_exp = Vec::with_capacity(by_pairs.len());
                for by_pair in by_pairs {
                    by_exp.push(parse_query_col_exp(by_pair, source_id)?)
                }
            }
            Rule::FromExp => {
                from_exp = parse_exp(some_pair.into_inner().next().unwrap(), source_id)?
            }
            Rule::WhereExp => {
                let where_pairs = some_pair.into_inner();
                where_exp = Vec::with_capacity(where_pairs.len());
                for where_pair in where_pairs {
                    where_exp.push(parse_exp(where_pair, source_id)?)
                }
            }
            Rule::LimitedExp => {
                limited_exp = Some(parse_exp(
                    some_pair.into_inner().next().unwrap(),
                    source_id,
                )?);
            }
            unexpected_exp => {
                return Err(raise_error(
                    format!("Unexpected query expression: {:?}", unexpected_exp),
                    some_pair.as_span(),
                ));
            }
        }
    }
    Ok(AstNode::Query {
        op,
        op_exp,
        by_exp,
        from_exp: Box::new(from_exp),
        where_exp,
        limited_exp: limited_exp.map(Box::new),
    })
}

fn parse_query_col_exp(pair: Pair<Rule>, source_id: usize) -> Result<AstNode, SpicyError> {
    match pair.as_rule() {
        Rule::ColExp => parse_query_col_exp(pair.into_inner().next().unwrap(), source_id),
        Rule::RenameColExp => {
            let mut pairs = pair.into_inner();
            let name = pairs.next().unwrap().as_str();
            let exp = parse_exp(pairs.next().unwrap(), source_id)?;
            Ok(AstNode::ColExp {
                name: name.to_owned(),
                exp: Box::new(exp),
            })
        }
        Rule::ColName => {
            let pair = pair.into_inner().next().unwrap();
            Ok(AstNode::Id {
                name: pair.as_str().to_owned(),
                pos: SourcePos::new(pair.as_span().start(), source_id),
            })
        }
        _ => parse_exp(pair, source_id),
    }
}

fn raise_error(msg: String, span: Span) -> SpicyError {
    SpicyError::from(PestError::new_from_span(
        ErrorVariant::CustomError { message: msg },
        span,
    ))
}

pub fn parse(source: &str, source_id: usize) -> Result<Vec<AstNode>, SpicyError> {
    let mut ast = vec![];
    // replace expected expression with syntax error
    let re = regex::Regex::new(r"= expected.+").unwrap();
    let pairs = ChiliParser::parse(Rule::Program, source)
        .map_err(|e| SpicyError::Err(re.replace(&e.to_string(), "= syntax error").to_string()))?;
    for pair in pairs {
        match pair.as_rule() {
            Rule::Exp
            | Rule::IfExp
            | Rule::TryExp
            | Rule::WhileExp
            | Rule::RaiseExp
            | Rule::ReturnExp => ast.push(parse_exp(pair, source_id)?),
            _ => {}
        }
    }
    Ok(ast)
}
