use std::fmt::Display;

use crate::errors::{SpicyError, SpicyResult};
use crate::obj::SpicyObj;

#[derive(PartialEq, Debug, Clone)]
pub struct SourcePos {
    pub source_id: usize,
    pub pos: usize,
}

impl SourcePos {
    pub fn new(pos: usize, source_id: usize) -> Self {
        Self { source_id, pos }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum AstNode {
    Show(Box<AstNode>),
    SpicyObj(SpicyObj),
    UnaryExp {
        f: Box<AstNode>,
        exp: Box<AstNode>,
    },
    BinaryExp {
        f2: Box<AstNode>,
        lhs: Box<AstNode>,
        rhs: Box<AstNode>,
    },
    AssignmentExp {
        id: String,
        exp: Box<AstNode>,
    },
    IndexAssignmentExp {
        id: String,
        indices: Vec<AstNode>,
        exp: Box<AstNode>,
    },
    Id {
        pos: SourcePos,
        name: String,
    },
    FnCall {
        pos: SourcePos,
        f: Box<AstNode>,
        args: Vec<AstNode>,
    },
    If {
        cond: Box<AstNode>,
        nodes: Vec<AstNode>,
    },
    While {
        cond: Box<AstNode>,
        nodes: Vec<AstNode>,
    },
    IfElse {
        nodes: Vec<AstNode>,
    },
    Try {
        tries: Vec<AstNode>,
        err_id: String,
        catches: Vec<AstNode>,
    },
    Return(Box<AstNode>),
    Raise(Box<AstNode>),
    Table(Vec<AstNode>),
    Matrix(Vec<AstNode>),
    Dict {
        keys: Vec<String>,
        values: Vec<AstNode>,
    },
    List(Vec<AstNode>),
    ColExp {
        name: String,
        exp: Box<AstNode>,
    },
    Query {
        op: QueryOp,
        op_exp: Vec<AstNode>,
        by_exp: Vec<AstNode>,
        from_exp: Box<AstNode>,
        where_exp: Vec<AstNode>,
        limited_exp: Option<Box<AstNode>>,
    },
    DelayedArg,
}

#[derive(PartialEq, Debug, Clone)]
pub enum QueryOp {
    Select,
    Update,
    Delete,
}

impl QueryOp {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "select" => Some(QueryOp::Select),
            "update" => Some(QueryOp::Update),
            "delete" => Some(QueryOp::Delete),
            _ => None,
        }
    }
}

impl AstNode {
    pub fn spicy_obj(self) -> SpicyResult<SpicyObj> {
        if let AstNode::SpicyObj(obj) = self {
            Ok(obj)
        } else {
            Err(SpicyError::ParserErr("Failed to cast to J".to_owned()))
        }
    }
    pub fn get_pos(&self) -> Option<SourcePos> {
        match self {
            AstNode::SpicyObj(obj) => {
                if let SpicyObj::Fn(func) = obj {
                    if func.is_built_in_fn() {
                        Some(func.pos.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            AstNode::Id { pos, name: _ } => Some(pos.clone()),
            AstNode::FnCall { pos, f: _, args: _ } => Some(pos.clone()),
            _ => None,
        }
    }
}

impl Display for AstNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AstNode::Show(_) => "show",
            AstNode::SpicyObj(obj) => return write!(f, "spicy obj - {}", obj),
            AstNode::UnaryExp { f: _, exp: _ } => "unary expression",
            AstNode::BinaryExp {
                f2: _,
                lhs: _,
                rhs: _,
            } => "binary expression",
            AstNode::AssignmentExp { id: _, exp: _ } => "assignment expression",
            AstNode::IndexAssignmentExp {
                id: _,
                indices: _,
                exp: _,
            } => "index assignment expression",
            AstNode::Id { name, .. } => name,
            AstNode::FnCall {
                pos: _,
                f: _,
                args: _,
            } => "function call",
            AstNode::Table(_) => "table expression",
            AstNode::Matrix(_) => "matrix expression",
            AstNode::Dict { keys: _, values: _ } => "dictionary expression",
            AstNode::List(_) => "list expression",
            AstNode::ColExp { name: _, exp: _ } => "column expression",
            AstNode::If { cond: _, nodes: _ } => "if expression",
            AstNode::While { cond: _, nodes: _ } => "while expression",
            AstNode::IfElse { nodes: _ } => "if else expression",
            AstNode::Try { .. } => "try expression",
            AstNode::Query { .. } => "query expression",
            AstNode::DelayedArg => "[::] delayed arg",
            AstNode::Return(_) => "return",
            AstNode::Raise(_) => "raise",
        };
        write!(f, "{}", s)
    }
}
