use std::fmt::Display;

use crate::{
    EngineState, Stack,
    ast_node::{AstNode, SourcePos},
    errors::SpicyError,
    obj::SpicyObj,
};

pub type FuncType = fn(&[&SpicyObj]) -> Result<SpicyObj, SpicyError>;
pub type SideEffectFuncType =
    fn(&EngineState, &mut Stack, &[&SpicyObj]) -> Result<SpicyObj, SpicyError>;

#[derive(PartialEq, Debug, Clone)]
pub struct Func {
    pub fn_body: String,
    pub pos: SourcePos,
    pub arg_num: usize,
    pub missing_index: Vec<usize>,
    pub params: Vec<String>,
    pub nodes: Box<Vec<AstNode>>,
    pub part_args: Option<Vec<SpicyObj>>,
    pub f: Option<Box<FuncType>>,
    pub f_with_side_effect: Option<Box<SideEffectFuncType>>,
    pub is_built_in_fn: bool,
    pub is_raw: bool,
}

impl Func {
    pub fn new(fn_body: &str, params: Vec<String>, nodes: Vec<AstNode>, pos: SourcePos) -> Self {
        Self {
            fn_body: fn_body.to_owned(),
            pos,
            arg_num: params.len(),
            missing_index: (0..params.len()).collect(),
            params,
            nodes: Box::new(nodes),
            part_args: None,
            f: None,
            f_with_side_effect: None,
            is_built_in_fn: false,
            is_raw: false,
        }
    }

    pub fn new_built_in_fn(
        f: Option<Box<FuncType>>,
        arg_num: usize,
        fn_body: &str,
        params: &[&str],
    ) -> Self {
        if params.len() != arg_num {
            panic!(
                "params length mismatch, fn: {}, params: {:?}, arg_num: {}",
                fn_body, params, arg_num
            );
        }
        Self {
            fn_body: fn_body.to_owned(),
            pos: SourcePos::new(0, 0),
            arg_num,
            missing_index: vec![],
            params: params.iter().map(|s| s.to_string()).collect(),
            nodes: Box::new(vec![]),
            part_args: None,
            f,
            f_with_side_effect: None,
            is_built_in_fn: true,
            is_raw: false,
        }
    }

    pub fn new_side_effect_built_in_fn(
        f_with_side_effect: Option<Box<SideEffectFuncType>>,
        arg_num: usize,
        fn_body: &str,
        params: &[&str],
    ) -> Self {
        Self {
            fn_body: fn_body.to_owned(),
            pos: SourcePos::new(0, 0),
            arg_num,
            missing_index: vec![],
            params: params.iter().map(|s| s.to_string()).collect(),
            nodes: Box::new(vec![]),
            part_args: None,
            f: None,
            f_with_side_effect,
            is_built_in_fn: true,
            is_raw: false,
        }
    }

    // need to parse fn body to get arg_num and params
    pub fn new_raw_fn(fn_body: &str) -> Self {
        Self {
            fn_body: fn_body.to_owned(),
            pos: SourcePos::new(0, 0),
            arg_num: 0,
            missing_index: vec![],
            params: vec![],
            nodes: Box::new(vec![]),
            part_args: None,
            f: None,
            f_with_side_effect: None,
            is_built_in_fn: false,
            is_raw: true,
        }
    }

    pub fn is_built_in_fn(&self) -> bool {
        self.is_built_in_fn
    }

    pub fn project(&self, args: &[&SpicyObj]) -> Self {
        let mut part_args = Vec::with_capacity(self.params.len());
        let mut missing_index: Vec<usize> = Vec::new();
        for i in 0..self.params.len() {
            if let Some(obj) = args.get(i) {
                part_args.push((*obj).clone());
                if obj.is_delayed_arg() {
                    missing_index.push(i);
                }
            } else {
                part_args.push(SpicyObj::DelayedArg);
                missing_index.push(i);
            }
        }
        let arg_num = missing_index.len();
        Self {
            part_args: Some(part_args),
            missing_index,
            arg_num,
            ..self.clone()
        }
    }

    pub fn is_side_effect(&self) -> bool {
        self.f_with_side_effect.is_some()
    }

    pub fn update_pos(&mut self, pos: SourcePos) {
        self.pos = pos;
    }
}

impl Display for Func {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let syntax = std::env::var("CHILI_SYNTAX").unwrap_or("chili".to_string());
        if syntax == "chili" {
            let fn_body = if self.is_built_in_fn() {
                format!("function({}){{}}", self.params.join(", "),)
            } else {
                self.fn_body.clone()
            };
            if self.part_args.is_none() {
                match &self.f {
                    Some(_) => write!(f, "function({}){{}}", self.params.join(", ")),
                    None => write!(f, "{}", fn_body),
                }
            } else {
                write!(
                    f,
                    "function({})\n{{\n  {}\n}}",
                    self.missing_index
                        .iter()
                        .map(|i| self.params[*i].clone())
                        .collect::<Vec<String>>()
                        .join(", "),
                    fn_body
                )
            }
        } else {
            let fn_body = if self.is_built_in_fn() {
                format!("{{[{}]}}", self.params.join("; "))
            } else {
                self.fn_body.clone()
            };
            if self.part_args.is_none() {
                match &self.f {
                    Some(_) => write!(f, "{{[{}]}}", self.params.join("; ")),
                    None => write!(f, "{}", fn_body),
                }
            } else {
                write!(
                    f,
                    "{{[{}]\n  {}\n}}",
                    self.missing_index
                        .iter()
                        .map(|i| self.params[*i].clone())
                        .collect::<Vec<String>>()
                        .join("; "),
                    fn_body
                )
            }
        }
    }
}
