use std::collections::HashMap;

use crate::{SpicyObj, errors::SpicyError, func::Func};

#[derive(Debug, Clone, PartialEq)]
pub struct Stack<'a> {
    pub vars: HashMap<String, SpicyObj>,
    pub f: Option<&'a Func>,
    pub src_path: Option<String>,
    pub stack_layer: usize,
    pub h: i64,
    pub user: String,
}

impl<'a> Default for Stack<'a> {
    fn default() -> Self {
        Self::new(None, 0, 0, "")
    }
}

impl<'a> Stack<'a> {
    pub fn new(src_path: Option<String>, stack_layer: usize, h: i64, user: &str) -> Self {
        Self {
            vars: HashMap::new(),
            f: None,
            src_path,
            stack_layer,
            h,
            user: user.to_owned(),
        }
    }

    pub fn get_var(&self, id: &str) -> Result<SpicyObj, SpicyError> {
        if id == "this.h" && self.h > 0 {
            Ok(SpicyObj::I64(self.h))
        } else if id == "this.user" && !self.user.is_empty() {
            Ok(SpicyObj::Symbol(self.user.clone()))
        } else {
            match self.vars.get(id) {
                Some(obj) => Ok(obj.clone()),
                None => Err(SpicyError::NameErr(id.to_owned())),
            }
        }
    }

    pub fn set_var(&mut self, id: &str, args: SpicyObj) {
        self.vars.insert(id.to_owned(), args);
    }

    pub fn clear_vars(&mut self) {
        self.vars.clear();
    }

    pub fn set_f(&mut self, f: &'a Func) {
        self.f = Some(f);
    }

    pub fn unset_f(&mut self) {
        self.f = None;
    }

    pub fn is_in_fn(&self) -> bool {
        self.f.is_some()
    }

    pub fn get_f(&self) -> Option<&'a Func> {
        self.f
    }

    pub fn get_base_path(&self) -> Option<String> {
        self.src_path.clone()
    }
}
