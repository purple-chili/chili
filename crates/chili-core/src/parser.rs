use std::num::ParseIntError;

use ariadne::{Report, ReportKind, Source};
use chili_parser::utils::get_err_msg;
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
use chili_parser::{Expr, Language, Span, Token};

use chumsky::prelude::*;

struct Context<'a> {
    pub source_id: usize,
    pub source_path: &'a str,
    pub source: &'a str,
    pub lang: Language,
}

impl<'a> Context<'a> {
    pub fn get_source_pos(&self, span: Span) -> SourcePos {
        SourcePos::new(span.start(), self.source_id)
    }

    pub fn get_source_code(&self, span: Span) -> &str {
        &self.source[span.start()..span.end()]
    }
}

fn parse_exp(expr: Expr, context: &Context) -> Result<AstNode, SpicyError> {
    match expr {
        Expr::Statement(statement) => parse_exp(statement.0, context),
        Expr::Unary { op, rhs, .. } => Ok(AstNode::UnaryExp {
            op: Box::new(parse_exp(*op, context)?),
            exp: Box::new(parse_exp(*rhs, context)?),
        }),
        Expr::Binary { lhs, op, rhs, .. } => Ok(AstNode::BinaryExp {
            op: Box::new(AstNode::Id {
                pos: context.get_source_pos(op.1),
                name: op.0.str().unwrap().to_owned(),
            }),
            lhs: Box::new(parse_exp(*lhs, context)?),
            rhs: Box::new(parse_exp(*rhs, context)?),
        }),
        Expr::Lit((token, span)) => parse_spicy_obj(token, span, context),
        Expr::Assign {
            id, indices, value, ..
        } => {
            if indices.is_empty() {
                Ok(AstNode::AssignmentExp {
                    id: id.id().unwrap(),
                    exp: Box::new(parse_exp(*value, context)?),
                })
            } else {
                let indices = indices
                    .into_iter()
                    .map(|index| parse_exp(index, context))
                    .collect::<Result<Vec<AstNode>, SpicyError>>()?;

                Ok(AstNode::IndexAssignmentExp {
                    id: id.id().unwrap(),
                    indices,
                    exp: Box::new(parse_exp(*value, context)?),
                })
            }
        }
        Expr::Id((token, span)) => Ok(AstNode::Id {
            name: token.str().unwrap().to_owned(),
            pos: context.get_source_pos(span),
        }),
        Expr::Fn { span, params, body } => {
            let params = params
                .iter()
                .map(|param| param.id().unwrap())
                .collect::<Vec<String>>();
            let nodes = body
                .block()
                .unwrap()
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            Ok(AstNode::SpicyObj(SpicyObj::Fn(Func::new(
                context.get_source_code(span),
                params,
                nodes,
                context.get_source_pos(span),
                context.lang.clone(),
            ))))
        }
        Expr::Call { span, f, args } => {
            let f = parse_exp(*f, context)?;
            let args = args
                .0
                .into_iter()
                .map(|arg| parse_exp(arg, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            Ok(AstNode::FnCall {
                pos: context.get_source_pos(span),
                f: Box::new(f),
                args,
                lang: context.lang.clone(),
            })
        }
        Expr::If {
            cond, then, else_, ..
        } => Ok(AstNode::If {
            cond: Box::new(parse_exp(*cond, context)?),
            nodes: then
                .block()
                .unwrap()
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?,
            else_nodes: Box::new(parse_exp(*else_, context)?),
        }),
        Expr::While { cond, body, .. } => Ok(AstNode::While {
            cond: Box::new(parse_exp(*cond, context)?),
            nodes: body
                .block()
                .unwrap()
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?,
        }),
        Expr::IfElse(if_else) => Ok(AstNode::IfElse {
            nodes: if_else
                .0
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?,
        }),
        Expr::Try {
            try_,
            err_id,
            catch,
            ..
        } => {
            let tries = try_
                .block()
                .unwrap()
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            let err_id = err_id.id().unwrap();
            let catches = catch
                .block()
                .unwrap()
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            Ok(AstNode::Try {
                tries,
                err_id,
                catches,
            })
        }
        Expr::Return(return_) => Ok(AstNode::Return(Box::new(parse_exp(return_.0, context)?))),
        Expr::Raise(raise) => Ok(AstNode::Raise(Box::new(parse_exp(raise.0, context)?))),
        Expr::DelayedArg(_) => Ok(AstNode::DelayedArg),
        Expr::DataFrame((dataframe, span)) => {
            let cols = dataframe
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            let is_all_series = cols
                .iter()
                .all(|col| matches!(col, AstNode::SpicyObj(SpicyObj::Series(_))));
            if is_all_series {
                let columns = cols
                    .into_iter()
                    .enumerate()
                    .map(|(i, col)| {
                        let mut column: Column =
                            col.spicy_obj().unwrap().series().unwrap().clone().into();
                        if column.name().is_empty() {
                            column.rename(format!("col{:02}", i).into());
                        }
                        column
                    })
                    .collect::<Vec<Column>>();
                let height = columns.first().map(|c| c.len()).unwrap_or(0);
                let df = DataFrame::new(height, columns)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                Ok(AstNode::SpicyObj(SpicyObj::DataFrame(df)))
            } else {
                Ok(AstNode::DataFrame(cols))
            }
        }
        Expr::Matrix((cols, span)) => {
            let cols = cols
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            let is_all_series = cols
                .iter()
                .all(|col| matches!(col, AstNode::SpicyObj(SpicyObj::Series(_))));
            if is_all_series {
                let columns = cols
                    .into_iter()
                    .enumerate()
                    .map(|(i, col)| {
                        let mut column: Column =
                            col.spicy_obj().unwrap().series().unwrap().clone().into();
                        if column.name().is_empty() {
                            column.rename(format!("col{:02}", i).into());
                        }
                        column
                    })
                    .collect::<Vec<Column>>();
                let height = columns.first().map(|c| c.len()).unwrap_or(0);
                let df = DataFrame::new(height, columns)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                let matrix = df
                    .to_ndarray::<Float64Type>(IndexOrder::C)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                Ok(AstNode::SpicyObj(SpicyObj::Matrix(
                    matrix.reversed_axes().to_shared(),
                )))
            } else {
                Ok(AstNode::Matrix(cols))
            }
        }
        Expr::Query {
            cmd,
            op,
            by,
            from,
            where_,
            limit,
            ..
        } => {
            let op = op
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            let by = by
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            let from = parse_exp(*from, context)?;
            let where_ = where_
                .into_iter()
                .map(|expr| parse_exp(expr, context))
                .collect::<Result<Vec<AstNode>, SpicyError>>()?;
            let limit = limit.map(|expr| parse_exp(expr, context)).transpose()?;
            Ok(AstNode::Query {
                op: QueryOp::from_str(&cmd.to_lowercase()).unwrap(),
                op_exp: op,
                by_exp: by,
                from_exp: Box::new(from),
                where_exp: where_,
                limited_exp: limit.map(Box::new),
            })
        }
        Expr::Bracket(bracket) => Ok(parse_exp(bracket.0, context)?),
        Expr::List((list, _)) => {
            if list.is_empty() {
                Ok(AstNode::SpicyObj(SpicyObj::MixedList(vec![])))
            } else {
                let list = list
                    .into_iter()
                    .map(|expr| parse_exp(expr, context))
                    .collect::<Result<Vec<AstNode>, SpicyError>>()?;
                Ok(AstNode::List(list))
            }
        }
        Expr::Dict((dict, _)) => {
            let mut keys = Vec::with_capacity(dict.len());
            let mut values = Vec::with_capacity(dict.len());
            for expr in dict {
                let (key, value) = expr.pair();
                keys.push(key.id().unwrap());
                let value = parse_exp(value, context)?;
                values.push(value);
            }
            Ok(AstNode::Dict { keys, values })
        }
        Expr::Pair { name, value } => {
            let name = name.id().unwrap();
            let value = parse_exp(value.0, context)?;
            Ok(AstNode::ColExp {
                name,
                exp: Box::new(value),
            })
        }
        Expr::Nil(_) => Ok(AstNode::SpicyObj(SpicyObj::Null)),
        Expr::Block((_, span)) | Expr::Error(span) => Err(raise_parser_error(
            format!("Unexpected expression: {:?}", expr),
            span,
            context,
        )),
    }
}

fn parse_spicy_obj(token: Token, span: Span, context: &Context) -> Result<AstNode, SpicyError> {
    let is_series = token
        .str()
        .map(|s| s.matches(' ').count() > 0)
        .unwrap_or(false);
    match token {
        Token::Null(_) => Ok(AstNode::SpicyObj(SpicyObj::Null)),
        Token::Column(name) => Ok(AstNode::SpicyObj(SpicyObj::Expr(col(name)))),
        Token::Bool(s) => match s.as_str() {
            "true" | "1b" => Ok(AstNode::SpicyObj(SpicyObj::Boolean(true))),
            "false" | "0b" => Ok(AstNode::SpicyObj(SpicyObj::Boolean(false))),
            _ => {
                let b: Vec<bool> = s
                    .as_bytes()
                    .iter()
                    .take(s.len() - 1)
                    .map(|u| *u == b'1')
                    .collect();
                let s = Series::new("".into(), b);
                Ok(AstNode::SpicyObj(SpicyObj::Series(s)))
            }
        },
        Token::Hex(s) => {
            if s.len() % 2 == 1 {
                Err(raise_parser_error(
                    format!("Invalid hex string: {:?}", s),
                    span,
                    context,
                ))
            } else if s.len() == 4 {
                Ok(AstNode::SpicyObj(SpicyObj::U8(
                    u8::from_str_radix(&s[2..], 16).unwrap(),
                )))
            } else {
                let s = &s[2..];
                match (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
                    .collect::<Result<Vec<u8>, ParseIntError>>()
                {
                    Ok(n) => Ok(AstNode::SpicyObj(SpicyObj::Series(Series::from_vec(
                        "".into(),
                        n,
                    )))),
                    Err(e) => Err(raise_parser_error(e.to_string(), span, context)),
                }
            }
        }
        Token::Date(s) => {
            if !is_series {
                let date = SpicyObj::parse_date(&s)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                Ok(AstNode::SpicyObj(date))
            } else {
                let dates = s
                    .split_whitespace()
                    .map(|s| {
                        SpicyObj::parse_date(s)
                            .map_err(|e| raise_parser_error(e.to_string(), span, context))
                    })
                    .collect::<Result<Vec<SpicyObj>, _>>()?
                    .iter()
                    .map(|args| args.to_i64().unwrap() as i32)
                    .collect::<Vec<i32>>();
                Ok(AstNode::SpicyObj(SpicyObj::Series(
                    Series::new("".into(), dates)
                        .cast(&PolarsDataType::Date)
                        .map_err(|e| raise_parser_error(e.to_string(), span, context))?,
                )))
            }
        }
        Token::Time(s) => {
            if !is_series {
                let time = SpicyObj::parse_time(&s)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                Ok(AstNode::SpicyObj(time))
            } else {
                let times = s
                    .split_whitespace()
                    .map(|s| {
                        SpicyObj::parse_time(s)
                            .map_err(|e| raise_parser_error(e.to_string(), span, context))
                    })
                    .collect::<Result<Vec<SpicyObj>, _>>()?
                    .iter()
                    .map(|args| args.to_i64().unwrap())
                    .collect::<Vec<i64>>();
                Ok(AstNode::SpicyObj(SpicyObj::Series(
                    Series::new("".into(), times)
                        .cast(&PolarsDataType::Time)
                        .map_err(|e| raise_parser_error(e.to_string(), span, context))?,
                )))
            }
        }
        Token::Datetime(s) => {
            if !is_series {
                let datetime = SpicyObj::parse_datetime(&s)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                Ok(AstNode::SpicyObj(datetime))
            } else {
                let datetimes = s
                    .split_whitespace()
                    .map(|s| {
                        SpicyObj::parse_datetime(s)
                            .map_err(|e| raise_parser_error(e.to_string(), span, context))
                    })
                    .collect::<Result<Vec<SpicyObj>, _>>()?
                    .iter()
                    .map(|args| args.to_i64().unwrap())
                    .collect::<Vec<i64>>();
                Ok(AstNode::SpicyObj(SpicyObj::Series(
                    Series::new("".into(), datetimes)
                        .cast(&PolarsDataType::Datetime(TimeUnit::Milliseconds, None))
                        .map_err(|e| raise_parser_error(e.to_string(), span, context))?,
                )))
            }
        }
        Token::Timestamp(s) => {
            if !is_series {
                let obj = SpicyObj::parse_timestamp(&s)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                Ok(AstNode::SpicyObj(obj))
            } else {
                let timestamps = s
                    .split_whitespace()
                    .map(|s| {
                        SpicyObj::parse_timestamp(s)
                            .map_err(|e| raise_parser_error(e.to_string(), span, context))
                    })
                    .collect::<Result<Vec<SpicyObj>, _>>()?
                    .iter()
                    .map(|args| args.to_i64().unwrap())
                    .collect::<Vec<i64>>();
                Ok(AstNode::SpicyObj(SpicyObj::Series(
                    Series::new("".into(), timestamps)
                        .cast(&PolarsDataType::Datetime(TimeUnit::Nanoseconds, None))
                        .map_err(|e| raise_parser_error(e.to_string(), span, context))?,
                )))
            }
        }
        Token::Duration(s) => {
            if !is_series {
                let obj = SpicyObj::parse_duration(&s)
                    .map_err(|e| raise_parser_error(e.to_string(), span, context))?;
                Ok(AstNode::SpicyObj(obj))
            } else {
                let durations = s
                    .split_whitespace()
                    .map(|s| {
                        SpicyObj::parse_duration(s)
                            .map_err(|e| raise_parser_error(e.to_string(), span, context))
                    })
                    .collect::<Result<Vec<SpicyObj>, _>>()?
                    .iter()
                    .map(|args| args.to_i64().unwrap())
                    .collect::<Vec<i64>>();
                Ok(AstNode::SpicyObj(SpicyObj::Series(
                    Series::new("".into(), durations)
                        .cast(&PolarsDataType::Duration(TimeUnit::Nanoseconds))
                        .map_err(|e| raise_parser_error(e.to_string(), span, context))?,
                )))
            }
        }
        Token::Int(s) => {
            // atom
            if s.matches(' ').count() == 0 {
                let integer = if s.ends_with('h') {
                    s[..s.len() - 1]
                        .parse::<i16>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::I16(n)))
                } else if s.ends_with('i') {
                    s[..s.len() - 1]
                        .parse::<i32>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::I32(n)))
                } else if s.ends_with("u8") {
                    s[..s.len() - 2]
                        .parse::<u8>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::U8(n)))
                } else if s.ends_with("i16") {
                    s[..s.len() - 3]
                        .parse::<i16>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::I16(n)))
                } else if s.ends_with("i32") {
                    s[..s.len() - 3]
                        .parse::<i32>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::I32(n)))
                } else if s.ends_with("i64") {
                    s[..s.len() - 3]
                        .parse::<i64>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::I64(n)))
                } else {
                    s.parse::<i64>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::I64(n)))
                };
                match integer {
                    Ok(integer) => Ok(integer),
                    Err(e) => Err(raise_parser_error(e.to_string(), span, context)),
                }
            } else {
                let obj = if let Some(numbers) = s.strip_suffix("h") {
                    SpicyObj::parse_numeric_series::<i16, Int16Type>(numbers)
                        .map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("i") {
                    SpicyObj::parse_numeric_series::<i32, Int32Type>(numbers)
                        .map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("u8") {
                    SpicyObj::parse_numeric_series::<u8, UInt8Type>(numbers)
                        .map_err(|e| e.to_string())
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
                    SpicyObj::parse_numeric_series::<i8, Int8Type>(numbers)
                        .map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("i16") {
                    SpicyObj::parse_numeric_series::<i16, Int16Type>(numbers)
                        .map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("i32") {
                    SpicyObj::parse_numeric_series::<i32, Int32Type>(numbers)
                        .map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("i64") {
                    SpicyObj::parse_numeric_series::<i64, Int64Type>(numbers)
                        .map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("i128") {
                    SpicyObj::parse_numeric_series::<i128, Int128Type>(numbers)
                        .map_err(|e| e.to_string())
                } else {
                    SpicyObj::parse_numeric_series::<i64, Int64Type>(&s).map_err(|e| e.to_string())
                };
                match obj {
                    Ok(obj) => Ok(AstNode::SpicyObj(obj)),
                    Err(e) => Err(raise_parser_error(e, span, context)),
                }
            }
        }
        Token::Float(s) => {
            if s.matches(' ').count() == 0 {
                let float = if s.ends_with("e") {
                    let s = &s[..s.len() - 1];
                    if s == "-0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F32(f32::NEG_INFINITY)))
                    } else if s == "0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F32(f32::INFINITY)))
                    } else {
                        s.parse::<f32>()
                            .map(|n| AstNode::SpicyObj(SpicyObj::F32(n)))
                    }
                } else if s.ends_with("f") {
                    let s = &s[..s.len() - 1];
                    if s == "-0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F64(f64::NEG_INFINITY)))
                    } else if s == "0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F64(f64::INFINITY)))
                    } else {
                        s.parse::<f64>()
                            .map(|n| AstNode::SpicyObj(SpicyObj::F64(n)))
                    }
                } else if s.ends_with("f32") {
                    let s = &s[..s.len() - 3];
                    if s == "-0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F32(f32::NEG_INFINITY)))
                    } else if s == "0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F32(f32::INFINITY)))
                    } else {
                        s.parse::<f32>()
                            .map(|n| AstNode::SpicyObj(SpicyObj::F32(n)))
                    }
                } else if s.ends_with("f64") {
                    let s = &s[..s.len() - 3];
                    if s == "-0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F64(f64::NEG_INFINITY)))
                    } else if s == "0w" {
                        Ok(AstNode::SpicyObj(SpicyObj::F64(f64::INFINITY)))
                    } else {
                        s.parse::<f64>()
                            .map(|n| AstNode::SpicyObj(SpicyObj::F64(n)))
                    }
                } else if s == "-0w" {
                    Ok(AstNode::SpicyObj(SpicyObj::F64(f64::NEG_INFINITY)))
                } else if s == "0w" {
                    Ok(AstNode::SpicyObj(SpicyObj::F64(f64::INFINITY)))
                } else {
                    s.parse::<f64>()
                        .map(|n| AstNode::SpicyObj(SpicyObj::F64(n)))
                };
                match float {
                    Ok(float) => Ok(float),
                    Err(e) => Err(raise_parser_error(e.to_string(), span, context)),
                }
            } else {
                let obj = if let Some(numbers) = s.strip_suffix("e") {
                    SpicyObj::parse_numeric_series_f32(numbers).map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("f") {
                    SpicyObj::parse_numeric_series_f64(numbers).map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("f32") {
                    SpicyObj::parse_numeric_series_f32(numbers).map_err(|e| e.to_string())
                } else if let Some(numbers) = s.strip_suffix("f64") {
                    SpicyObj::parse_numeric_series_f64(numbers).map_err(|e| e.to_string())
                } else {
                    SpicyObj::parse_numeric_series_f64(&s).map_err(|e| e.to_string())
                };
                match obj {
                    Ok(obj) => Ok(AstNode::SpicyObj(obj)),
                    Err(e) => Err(raise_parser_error(e, span, context)),
                }
            }
        }
        Token::Symbol(s) => {
            if s.matches('`').count() == 1 {
                Ok(AstNode::SpicyObj(SpicyObj::Symbol(s[1..].to_string())))
            } else {
                Ok(AstNode::SpicyObj(SpicyObj::Series(
                    Series::new("".into(), s[1..].split('`').collect::<Vec<_>>())
                        .cast(&PolarsDataType::Categorical(
                            Categories::global(),
                            Categories::global().mapping(),
                        ))
                        .unwrap(),
                )))
            }
        }
        Token::Str(s) => Ok(AstNode::SpicyObj(SpicyObj::String(s))),
        unexpected_rule => Err(raise_parser_error(
            format!("Unexpected rule for spicy obj: {:?}", unexpected_rule),
            span,
            context,
        )),
    }
}

fn raise_parser_error(msg: String, span: Span, context: &Context) -> SpicyError {
    let report = Report::build(ReportKind::Error, (context.source_path, span.into_range()))
        .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
        .with_message(msg)
        .finish();
    let mut buf = Vec::new();
    report
        .write_for_stdout(
            (context.source_path, Source::from(context.source)),
            &mut buf,
        )
        .unwrap();
    let err_msg = String::from_utf8(buf).unwrap();
    SpicyError::ParserErr(err_msg)
}

pub fn parse(
    source: &str,
    source_id: usize,
    source_path: &str,
) -> Result<Vec<AstNode>, SpicyError> {
    let mut ast = vec![];
    // replace expected expression with syntax error
    let re = regex::Regex::new(r"expected.+").unwrap();
    let (tokens, errs) = Token::lexer().parse(source).into_output_errors();
    if !errs.is_empty() {
        return Err(SpicyError::Err(get_err_msg(errs, source_path, source)));
    }

    let tokens = tokens
        .unwrap()
        .into_iter()
        .filter(|(t, _)| !matches!(t, Token::Comment(_)))
        .collect::<Vec<_>>();

    let mut lang = Language::Chili;

    let (program, errs) = if source_path.ends_with(".chi") {
        Expr::parser_chili()
            .parse(
                tokens
                    .as_slice()
                    .map((source.len()..source.len()).into(), |(t, s)| (t, s)),
            )
            .into_output_errors()
    } else {
        lang = Language::Pepper;
        Expr::parser_pepper()
            .parse(
                tokens
                    .as_slice()
                    .map((source.len()..source.len()).into(), |(t, s)| (t, s)),
            )
            .into_output_errors()
    };

    if !errs.is_empty() {
        let errs = get_err_msg(errs, source_path, source);
        return Err(SpicyError::Err(
            re.replace(&errs, "syntax error").to_string(),
        ));
    }

    let program = program.unwrap();

    let context = Context {
        source_id,
        source_path,
        source,
        lang,
    };
    if let Some(block) = program.block() {
        for expr in block {
            ast.push(parse_exp(expr, &context)?);
        }
    }

    Ok(ast)
}
