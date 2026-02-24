use crate::token::{Span, Token};
use chumsky::{input::ValueInput, prelude::*};

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit((Token, Span)),
    // empty expression
    Nil(Span),
    DelayedArg(Span),
    Block((Vec<Self>, Span)),
    Statement(Box<(Self, Span)>),
    Error(Span),
    Unary {
        span: Span,
        op: Box<Self>,
        rhs: Box<Self>,
    },
    Binary {
        span: Span,
        lhs: Box<Self>,
        op: (Token, Span),
        rhs: Box<Self>,
    },
    Assign {
        span: Span,
        id: Box<Self>,
        indices: Vec<Self>,
        value: Box<Self>,
    },
    Id((Token, Span)),
    Call {
        span: Span,
        f: Box<Self>,
        args: (Vec<Self>, Span),
    },
    If {
        span: Span,
        cond: Box<Self>,
        then: Box<Self>,
        else_: Box<Self>,
    },
    While {
        span: Span,
        cond: Box<Self>,
        body: Box<Self>,
    },
    IfElse((Vec<Self>, Span)),
    Try {
        span: Span,
        try_: Box<Self>,
        err_id: Box<Self>,
        catch: Box<Self>,
    },
    Return(Box<(Self, Span)>),
    Raise(Box<(Self, Span)>),
    Bracket(Box<(Self, Span)>),
    DataFrame((Vec<Self>, Span)),
    Matrix((Vec<Self>, Span)),
    Dict((Vec<Self>, Span)),
    List((Vec<Self>, Span)),
    Pair {
        name: Box<Self>,
        value: Box<(Self, Span)>,
    },
    Query {
        span: Span,
        cmd: String,
        op: Vec<Self>,
        by: Vec<Self>,
        from: Box<Self>,
        where_: Vec<Self>,
        limit: Box<Option<Self>>,
    },
    Fn {
        span: Span,
        params: Vec<Self>,
        body: Box<Self>,
    },
}

impl Expr {
    pub fn parser_chili<'a, I>()
    -> impl Parser<'a, I, Expr, extra::Err<Rich<'a, Token, Span>>> + Clone
    where
        I: ValueInput<'a, Token = Token, Span = Span>,
    {
        let list_open = Token::Punc('[');
        let list_close = Token::Punc(']');
        let args_open = Token::Punc('(');
        let args_close = Token::Punc(')');

        let id = select! { Token::Id(id) = e => Expr::Id((Token::Id(id), e.span())) }
            .labelled("identifier");

        let statement = recursive(|statement| {
            let inline_expr = recursive(|inline_expr| {
                let block = just(Token::Punc('{'))
                    .ignore_then(statement.clone().repeated().collect::<Vec<_>>())
                    .then_ignore(just(Token::Punc('}')))
                    .map_with(|mut exprs, e| {
                        let is_statement = exprs
                            .last()
                            .map(|expr| matches!(expr, Expr::Statement(_)))
                            .unwrap_or(false);

                        let span: Span = e.span();
                        if !is_statement {
                            let span_end = span.end - 1;
                            exprs.push(Expr::Nil((span_end..span_end).into()));
                        }

                        Expr::Block((exprs, span))
                    })
                    .boxed();

                let lit = select! {
                    Token::Null(n) = e => Expr::Lit((Token::Null(n), e.span())),
                    Token::Bool(b) = e => Expr::Lit((Token::Bool(b), e.span())),
                    Token::Hex(h) = e => Expr::Lit((Token::Hex(h), e.span())),
                    Token::Timestamp(t) = e => Expr::Lit((Token::Timestamp(t), e.span())),
                    Token::Datetime(d) = e => Expr::Lit((Token::Datetime(d), e.span())),
                    Token::Duration(d) = e => Expr::Lit((Token::Duration(d), e.span())),
                    Token::Date(d) = e => Expr::Lit((Token::Date(d), e.span())),
                    Token::Time(t) = e => Expr::Lit((Token::Time(t), e.span())),
                    Token::Int(i) = e => Expr::Lit((Token::Int(i), e.span())),
                    Token::Float(f) = e => Expr::Lit((Token::Float(f), e.span())),
                    Token::Symbol(s) = e => Expr::Lit((Token::Symbol(s), e.span())),
                    Token::Str(s) = e => Expr::Lit((Token::Str(s), e.span())),
                    Token::Column(c) = e => Expr::Lit((Token::Column(c), e.span())),
                }
                .labelled("literal");

                let op =
                    select! { Token::Op(op) = e => (Token::Op(op), e.span()) }.labelled("operator");

                let op_as_id = select! { Token::Op(op) = e => Expr::Id((Token::Op(op), e.span())) };

                let list_exprs = inline_expr
                    .clone()
                    .or(op_as_id)
                    .separated_by(just(Token::Punc(',')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let list = list_exprs
                    .delimited_by(just(list_open), just(list_close))
                    .map_with(|v, e| Expr::List((v, e.span())))
                    .labelled("list")
                    .boxed();

                let indices = inline_expr
                    .clone()
                    .separated_by(just(Token::Punc(',')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(args_open.clone()), just(args_close.clone()))
                    .validate(|indices, e, emitter| {
                        if indices.is_empty() {
                            emitter.emit(Rich::custom(
                                e.span(),
                                "required at least one index for indices assignment",
                            ));
                        }
                        indices
                    })
                    .labelled("indices")
                    .boxed();

                let id_with_indices = id
                    .then(indices.clone().map_with(|v, e| (v, e.span())))
                    .map_with(|(id, indices), e| Expr::Call {
                        span: e.span(),
                        f: id.boxed(),
                        args: indices,
                    });

                let assign = id_with_indices
                    .or(id)
                    .then_ignore(just(Token::Op(":".to_string())))
                    .then(inline_expr.clone())
                    .map_with(|(id, value), e| {
                        let (id, indices) = if let Expr::Call { f, args, .. } = id {
                            (f, args.0)
                        } else {
                            (id.boxed(), (vec![]))
                        };
                        Expr::Assign {
                            span: e.span(),
                            id,
                            indices,
                            value: value.boxed(),
                        }
                    })
                    .labelled("assignment")
                    .as_context()
                    .boxed();

                let pair = id
                    .then_ignore(just(Token::Op(":".to_string())))
                    .then(inline_expr.clone().map_with(|e, s| (e, s.span())))
                    .map(|(name, value)| Expr::Pair {
                        name: name.boxed(),
                        value: Box::new(value),
                    })
                    .boxed();

                let pairs = pair
                    .clone()
                    .separated_by(just(Token::Punc(',')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let dict = pairs
                    .clone()
                    .delimited_by(just(Token::Punc('{')), just(Token::Punc('}')))
                    .map_with(|v, e| Expr::Dict((v, e.span())))
                    .labelled("dictionary")
                    .boxed();

                // expr or operator
                let bracket = inline_expr
                    .clone()
                    .or(op_as_id)
                    .delimited_by(just(Token::Punc('(')), just(Token::Punc(')')))
                    .map_with(|ex, e| Expr::Bracket(Box::new((ex, e.span()))))
                    .labelled("bracket")
                    .boxed();

                let params = id
                    .or(op.map(Expr::Id))
                    .separated_by(just(Token::Punc(',')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(args_open.clone()), just(args_close.clone()))
                    .validate(|params, _, emitter| {
                        for param in params.iter() {
                            if let Expr::Id(id) = param {
                                let (token, span) = id;
                                if token.is_operator() {
                                    emitter.emit(Rich::custom(
                                        *span,
                                        "built-in functions cannot be used as parameter name",
                                    ));
                                }
                            }
                        }
                        params
                    })
                    .labelled("function params")
                    .boxed();

                let fn_ = just(Token::Fn)
                    .then(params)
                    .then(block.clone())
                    .map_with(|(params, block), e| Expr::Fn {
                        span: e.span(),
                        params: params.1,
                        body: block.boxed(),
                    })
                    .labelled("function")
                    .boxed();

                let column = pair.clone().or(inline_expr.clone());

                let columns = column
                    .clone()
                    .separated_by(just(Token::Punc(',')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let df = just(Token::Punc('('))
                    .then(just(Token::Punc('[')))
                    .then(just(Token::Punc(']')))
                    .ignore_then(columns.clone())
                    .then_ignore(just(Token::Punc(')')))
                    .map_with(|v, e| Expr::DataFrame((v, e.span())))
                    .labelled("dataframe")
                    .boxed();

                let matrix_exprs = inline_expr
                    .clone()
                    .separated_by(just(Token::Punc(',')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let matrix = just(Token::Punc('['))
                    .then(just(Token::Punc('[')))
                    .ignore_then(matrix_exprs.clone())
                    .then_ignore(just(Token::Punc(']')))
                    .then_ignore(just(Token::Punc(']')))
                    .map_with(|v, e| Expr::Matrix((v, e.span())))
                    .labelled("matrix")
                    .boxed();

                let term = choice((
                    df.clone(),
                    matrix.clone(),
                    dict.clone(),
                    list.clone(),
                    lit,
                    id,
                ));

                let args = inline_expr
                    .clone()
                    .or_not()
                    .map_with(|arg, e| arg.unwrap_or(Expr::DelayedArg(e.span())))
                    .separated_by(just(Token::Punc(',')))
                    .collect::<Vec<_>>()
                    .delimited_by(just(args_open.clone()), just(args_close.clone()))
                    .labelled("arguments")
                    .boxed();

                let call = choice((
                    id,
                    fn_.clone(),
                    op_as_id,
                    df.clone(),
                    list.clone(),
                    bracket.clone(),
                ))
                .then(args.map_with(|v, e| (v, e.span())))
                .map_with(|(f, args), e| {
                    if args.0.len() == 1 && args.0[0].is_delayed_arg() {
                        Expr::Call {
                            span: e.span(),
                            f: Box::new(f),
                            args: (vec![], args.0[0].span()),
                        }
                    } else {
                        Expr::Call {
                            span: e.span(),
                            f: Box::new(f),
                            args,
                        }
                    }
                })
                .labelled("call")
                .boxed();

                let where_ = inline_expr
                    .clone()
                    .separated_by(just(Token::Punc(',')))
                    .collect::<Vec<_>>();

                let select = just(Token::Select)
                    .or(just(Token::Update))
                    .then(columns.clone())
                    .then(just(Token::By).ignore_then(columns.clone()).or_not())
                    .then(just(Token::From).ignore_then(inline_expr.clone()))
                    .then(just(Token::Where).ignore_then(where_.clone()).or_not())
                    .then(just(Token::Limit).ignore_then(inline_expr.clone()).or_not())
                    .map_with(
                        |(((((op, columns), by), from), where_), limit), e| Expr::Query {
                            span: e.span(),
                            cmd: op.to_string(),
                            op: columns,
                            by: by.unwrap_or(vec![]),
                            from: from.boxed(),
                            where_: where_.unwrap_or(vec![]),
                            limit: Box::new(limit),
                        },
                    )
                    .boxed();

                let delete0 = just(Token::Delete)
                    .then(just(Token::From).ignore_then(inline_expr.clone()))
                    .then(just(Token::Where).ignore_then(where_.clone()).or_not())
                    .map_with(|((cmd, from), where_), e| Expr::Query {
                        span: e.span(),
                        cmd: cmd.to_string(),
                        op: vec![],
                        by: vec![],
                        from: from.boxed(),
                        where_: where_.unwrap_or(vec![]),
                        limit: Box::new(None),
                    })
                    .boxed();

                let delete1 = just(Token::Delete)
                    .then(columns.clone())
                    .then(just(Token::From).ignore_then(inline_expr.clone()))
                    .map_with(|((cmd, columns), from), e| Expr::Query {
                        span: e.span(),
                        cmd: cmd.to_string(),
                        op: columns,
                        by: vec![],
                        from: from.boxed(),
                        where_: vec![],
                        limit: Box::new(None),
                    })
                    .boxed();

                let query = choice((select.clone(), delete0.clone(), delete1.clone()))
                    .labelled("query")
                    .as_context()
                    .boxed();

                let if_else = just(Token::If)
                    .ignore_then(block.clone())
                    .map_with(|v, e| Expr::IfElse((v.block().unwrap(), e.span())));

                let operand = choice((
                    query.clone(),
                    if_else.clone(),
                    fn_.clone(),
                    call.clone(),
                    bracket.clone(),
                    term.clone(),
                ))
                .labelled("operand")
                .boxed();

                let binary = operand
                    .clone()
                    .foldl_with(op.then(operand).repeated().at_least(1), |a, (op, b), e| {
                        Expr::Binary {
                            span: e.span(),
                            lhs: Box::new(a),
                            op,
                            rhs: Box::new(b),
                        }
                    })
                    .labelled("binary")
                    .as_context()
                    .boxed();

                choice((
                    query.clone(),
                    if_else.clone(),
                    fn_.clone(),
                    assign.clone(),
                    binary.clone(),
                    call.clone(),
                    bracket.clone(),
                    term.clone(),
                ))
                .labelled("inline_expr")
                .boxed()
                // end of inline_expr
            });

            let block = just(Token::Punc('{'))
                .ignore_then(statement.clone().repeated().collect::<Vec<_>>())
                .then_ignore(just(Token::Punc('}')))
                .map_with(|mut exprs, e| {
                    let is_statement = exprs
                        .last()
                        .map(|expr| matches!(expr, Expr::Statement(_)))
                        .unwrap_or(false);

                    let span: Span = e.span();
                    if !is_statement {
                        let span_end = span.end - 1;
                        exprs.push(Expr::Nil((span_end..span_end).into()));
                    }

                    Expr::Block((exprs, span))
                })
                .boxed();

            let if_ = recursive(|if_| {
                just(Token::If)
                    .ignore_then(
                        just(Token::Punc('('))
                            .ignore_then(inline_expr.clone())
                            .then_ignore(just(Token::Punc(')'))),
                    )
                    .then(block.clone())
                    .then(
                        just(Token::Else)
                            .ignore_then(block.clone().or(if_))
                            .or_not(),
                    )
                    .map_with(|((cond, then), else_), e| {
                        let span: Span = e.span();
                        Expr::If {
                            span,
                            cond: Box::new(cond),
                            then: Box::new(then),
                            else_: Box::new(
                                else_.unwrap_or(Expr::Nil((span.end..span.end).into())),
                            ),
                        }
                    })
            })
            .boxed();

            let while_ = just(Token::While)
                .ignore_then(
                    just(Token::Punc('('))
                        .ignore_then(inline_expr.clone())
                        .then_ignore(just(Token::Punc(')'))),
                )
                .then(block.clone())
                .map_with(|(cond, body), e| Expr::While {
                    span: e.span(),
                    cond: Box::new(cond),
                    body: Box::new(body),
                })
                .labelled("while")
                .boxed();

            let try_ = just(Token::Try)
                .ignore_then(block.clone())
                .then_ignore(just(Token::Catch))
                .then(
                    just(Token::Punc('('))
                        .ignore_then(id)
                        .then_ignore(just(Token::Punc(')'))),
                )
                .then(block.clone())
                .map_with(|((tries, id), catches), e| Expr::Try {
                    span: e.span(),
                    try_: Box::new(tries),
                    err_id: Box::new(id),
                    catch: Box::new(catches),
                })
                .labelled("try")
                .boxed();

            let return_ = just(Token::Return)
                .ignore_then(inline_expr.clone())
                .map_with(|expr, e| Expr::Return(Box::new((expr, e.span()))));

            let raise_ = just(Token::Raise)
                .ignore_then(inline_expr.clone())
                .map_with(|expr, e| Expr::Raise(Box::new((expr, e.span()))));

            let unterminated_statement =
                choice((inline_expr.clone(), return_.clone(), raise_.clone()))
                    .map_with(|s, e| Expr::Statement(Box::new((s, e.span()))))
                    .boxed();

            let terminated_statement = inline_expr
                .clone()
                .or(return_)
                .or(raise_)
                .clone()
                .then_ignore(just(Token::Punc(';')));

            choice((
                if_.clone(),
                while_.clone(),
                try_.clone(),
                terminated_statement.clone(),
                unterminated_statement.clone(),
            ))
            .labelled("statement")
            .boxed()
        });

        statement
            .clone()
            .repeated()
            .at_least(1)
            .collect::<Vec<_>>()
            .map_with(|exprs, e| Expr::Block((exprs, e.span())))
            .labelled("statements")
    }

    pub fn parser_pepper<'a, I>()
    -> impl Parser<'a, I, Expr, extra::Err<Rich<'a, Token, Span>>> + Clone
    where
        I: ValueInput<'a, Token = Token, Span = Span>,
    {
        let list_open = Token::Punc('(');
        let list_close = Token::Punc(')');
        let args_open = Token::Punc('[');
        let args_close = Token::Punc(']');

        let id = select! { Token::Id(id) = e => Expr::Id((Token::Id(id), e.span())) }
            .labelled("identifier");

        let statement = recursive(|statement| {
            let inline_expr = recursive(|inline_expr| {
                let block = statement
                    .clone()
                    .or_not()
                    .map_with(|expr, e| expr.unwrap_or(Expr::Nil(e.span())))
                    .separated_by(just(Token::Punc(';')))
                    .collect::<Vec<_>>()
                    .delimited_by(just(args_open.clone()), just(args_close.clone()))
                    .map_with(|exprs, e| (exprs, e.span()))
                    .boxed();

                let lit = select! {
                    Token::Null(n) = e => Expr::Lit((Token::Null(n), e.span())),
                    Token::Bool(b) = e => Expr::Lit((Token::Bool(b), e.span())),
                    Token::Hex(h) = e => Expr::Lit((Token::Hex(h), e.span())),
                    Token::Timestamp(t) = e => Expr::Lit((Token::Timestamp(t), e.span())),
                    Token::Datetime(d) = e => Expr::Lit((Token::Datetime(d), e.span())),
                    Token::Duration(d) = e => Expr::Lit((Token::Duration(d), e.span())),
                    Token::Date(d) = e => Expr::Lit((Token::Date(d), e.span())),
                    Token::Time(t) = e => Expr::Lit((Token::Time(t), e.span())),
                    Token::Int(i) = e => Expr::Lit((Token::Int(i), e.span())),
                    Token::Float(f) = e => Expr::Lit((Token::Float(f), e.span())),
                    Token::Symbol(s) = e => Expr::Lit((Token::Symbol(s), e.span())),
                    Token::Str(s) = e => Expr::Lit((Token::Str(s), e.span())),
                    Token::Column(c) = e => Expr::Lit((Token::Column(c), e.span())),
                }
                .labelled("literal");

                let op =
                    select! { Token::Op(op) = e => (Token::Op(op), e.span()) }.labelled("operator");

                let op_as_id = select! { Token::Op(op) = e => Expr::Id((Token::Op(op), e.span())) };

                // A list of expressions used in list, matrix
                let list_exprs = inline_expr
                    .clone()
                    .or(op_as_id)
                    .separated_by(just(Token::Punc(';')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let list = list_exprs
                    .delimited_by(just(list_open), just(list_close))
                    .map_with(|v, e| Expr::List((v, e.span())))
                    .labelled("list")
                    .boxed();

                let indices = inline_expr
                    .clone()
                    .separated_by(just(Token::Punc(';')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(args_open.clone()), just(args_close.clone()))
                    .validate(|indices, e, emitter| {
                        if indices.is_empty() {
                            emitter.emit(Rich::custom(
                                e.span(),
                                "required at least one index for indices assignment",
                            ));
                        }
                        indices
                    })
                    .labelled("indices")
                    .boxed();

                let id_with_indices = id
                    .then(indices.clone().map_with(|v, e| (v, e.span())))
                    .map_with(|(id, indices), e| Expr::Call {
                        span: e.span(),
                        f: id.boxed(),
                        args: indices,
                    });

                let assign = id_with_indices
                    .or(id)
                    .then_ignore(just(Token::Op(":".to_string())))
                    .then(inline_expr.clone())
                    .map_with(|(id, value), e| {
                        let (id, indices) = if let Expr::Call { f, args, .. } = id {
                            (f, args.0)
                        } else {
                            (id.boxed(), vec![])
                        };
                        Expr::Assign {
                            span: e.span(),
                            id,
                            indices,
                            value: value.boxed(),
                        }
                    })
                    .labelled("assignment")
                    .as_context()
                    .boxed();

                let pair = id
                    .then_ignore(just(Token::Op(":".to_string())))
                    .then(inline_expr.clone().map_with(|e, s| (e, s.span())))
                    .map(|(name, value)| Expr::Pair {
                        name: name.boxed(),
                        value: Box::new(value),
                    })
                    .boxed();

                let pairs = pair
                    .clone()
                    .separated_by(just(Token::Punc(';')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let dict = pairs
                    .clone()
                    .delimited_by(just(Token::Punc('{')), just(Token::Punc('}')))
                    .map_with(|v, e| Expr::Dict((v, e.span())))
                    .labelled("dictionary")
                    .boxed();

                // expr or operator
                let bracket = inline_expr
                    .clone()
                    .or(op_as_id)
                    .delimited_by(just(Token::Punc('(')), just(Token::Punc(')')))
                    .map_with(|ex, e| Expr::Bracket(Box::new((ex, e.span()))))
                    .labelled("bracket")
                    .boxed();

                let params = id
                    .or(op.map(Expr::Id))
                    .separated_by(just(Token::Punc(';')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(args_open.clone()), just(args_close.clone()))
                    .validate(|params, _, emitter| {
                        for param in params.iter() {
                            if let Expr::Id(id) = param {
                                let (token, span) = id;
                                if token.is_operator() {
                                    emitter.emit(Rich::custom(
                                        *span,
                                        "built-in functions cannot be used as parameter name",
                                    ));
                                }
                            }
                        }
                        params
                    })
                    .labelled("function params")
                    .boxed();

                let fn_ = just(Token::Punc('{'))
                    .ignore_then(params)
                    .then(
                        statement
                            .clone()
                            .or_not()
                            .map_with(|expr, e| expr.unwrap_or(Expr::Nil(e.span())))
                            .separated_by(just(Token::Punc(';')))
                            .collect::<Vec<_>>()
                            .map_with(|exprs, e| Expr::Block((exprs, e.span()))),
                    )
                    .then_ignore(just(Token::Punc('}')))
                    .map_with(|(params, block), e| Expr::Fn {
                        span: e.span(),
                        params,
                        body: block.boxed(),
                    })
                    .labelled("function")
                    .boxed();

                let df_column = pair.clone().or(inline_expr.clone());

                let df_columns = df_column
                    .clone()
                    .separated_by(just(Token::Punc(';')))
                    .allow_trailing()
                    .collect::<Vec<_>>();

                let df = just(Token::Punc('('))
                    .then(just(Token::Punc('[')))
                    .then(just(Token::Punc(']')))
                    .ignore_then(df_columns.clone())
                    .then_ignore(just(Token::Punc(')')))
                    .map_with(|v, e| Expr::DataFrame((v, e.span())))
                    .labelled("dataframe")
                    .boxed();

                let matrix_exprs_exprs = inline_expr
                    .clone()
                    .separated_by(just(Token::Punc(';')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let matrix = just(Token::Punc('['))
                    .then(just(Token::Punc('[')))
                    .ignore_then(matrix_exprs_exprs.clone())
                    .then_ignore(just(Token::Punc(']')))
                    .then_ignore(just(Token::Punc(']')))
                    .map_with(|v, e| Expr::Matrix((v, e.span())))
                    .labelled("matrix")
                    .boxed();

                let term = df
                    .clone()
                    .or(matrix.clone())
                    .or(dict.clone())
                    .or(list.clone())
                    .or(lit)
                    .or(id)
                    .boxed();

                let args = inline_expr
                    .clone()
                    .or_not()
                    .map_with(|arg, e| arg.unwrap_or(Expr::DelayedArg(e.span())))
                    .separated_by(just(Token::Punc(';')))
                    .collect::<Vec<_>>()
                    .delimited_by(just(args_open.clone()), just(args_close.clone()))
                    .labelled("arguments")
                    .boxed();

                let call = id
                    .or(fn_.clone())
                    .or(op_as_id)
                    .or(df.clone())
                    .or(list.clone())
                    .or(bracket.clone())
                    .then(args.map_with(|v, e| (v, e.span())))
                    .map_with(|(f, args), e| {
                        if args.0.len() == 1 && args.0[0].is_delayed_arg() {
                            Expr::Call {
                                span: e.span(),
                                f: Box::new(f),
                                args: (vec![], args.0[0].span()),
                            }
                        } else {
                            Expr::Call {
                                span: e.span(),
                                f: Box::new(f),
                                args,
                            }
                        }
                    })
                    .labelled("call")
                    .boxed();

                let where_ = inline_expr
                    .clone()
                    .separated_by(just(Token::Punc(',')))
                    .collect::<Vec<_>>();

                let column = pair.clone().or(inline_expr.clone());

                let columns = column
                    .clone()
                    .separated_by(just(Token::Punc(',')))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .boxed();

                let select = just(Token::Select)
                    .or(just(Token::Update))
                    .then(columns.clone())
                    .then(just(Token::By).ignore_then(columns.clone()).or_not())
                    .then(just(Token::From).ignore_then(inline_expr.clone()))
                    .then(just(Token::Where).ignore_then(where_.clone()).or_not())
                    .then(just(Token::Limit).ignore_then(inline_expr.clone()).or_not())
                    .map_with(
                        |(((((op, columns), by), from), where_), limit), e| Expr::Query {
                            span: e.span(),
                            cmd: op.to_string(),
                            op: columns,
                            by: by.unwrap_or(vec![]),
                            from: from.boxed(),
                            where_: where_.unwrap_or(vec![]),
                            limit: Box::new(limit),
                        },
                    )
                    .boxed();

                let delete0 = just(Token::Delete)
                    .then(just(Token::From).ignore_then(inline_expr.clone()))
                    .then(just(Token::Where).ignore_then(where_.clone()).or_not())
                    .map_with(|((cmd, from), where_), e| Expr::Query {
                        span: e.span(),
                        cmd: cmd.to_string(),
                        op: vec![],
                        by: vec![],
                        from: from.boxed(),
                        where_: where_.unwrap_or(vec![]),
                        limit: Box::new(None),
                    })
                    .boxed();

                let delete1 = just(Token::Delete)
                    .then(columns.clone())
                    .then(just(Token::From).ignore_then(inline_expr.clone()))
                    .map_with(|((cmd, columns), from), e| Expr::Query {
                        span: e.span(),
                        cmd: cmd.to_string(),
                        op: columns,
                        by: vec![],
                        from: from.boxed(),
                        where_: vec![],
                        limit: Box::new(None),
                    })
                    .boxed();

                let query = select
                    .or(delete0)
                    .or(delete1)
                    .labelled("query")
                    .as_context()
                    .boxed();

                let if_else = just(Token::Op("$".to_string()))
                    .ignore_then(block.clone())
                    .map_with(|v, e| Expr::IfElse((v.0, e.span())));

                let operand = choice((
                    if_else.clone(),
                    call.clone(),
                    fn_.clone(),
                    bracket.clone(),
                    term.clone(),
                ));

                let binary = operand
                    .clone()
                    .then(op)
                    .repeated()
                    .at_least(1)
                    .foldr_with(inline_expr.clone(), |(a, op), b, e| Expr::Binary {
                        span: e.span(),
                        lhs: Box::new(a),
                        op,
                        rhs: Box::new(b),
                    })
                    .labelled("binary")
                    .as_context()
                    .boxed();

                let unary = operand
                    .then(inline_expr.clone())
                    .map_with(|(op, rhs), e| Expr::Unary {
                        span: e.span(),
                        op: op.boxed(),
                        rhs: rhs.boxed(),
                    })
                    .labelled("unary")
                    .as_context()
                    .boxed();

                choice((
                    query.clone(),
                    if_else.clone(),
                    assign.clone(),
                    binary.clone(),
                    unary.clone(),
                    call.clone(),
                    fn_.clone(),
                    bracket.clone(),
                    term.clone(),
                ))
                .labelled("inline_expr")
                .boxed()
                // end of inline_expr
            });

            let block = statement
                .clone()
                .or_not()
                .map_with(|expr, e| expr.unwrap_or(Expr::Nil(e.span())))
                .separated_by(just(Token::Punc(';')))
                .collect::<Vec<_>>()
                .delimited_by(just(args_open.clone()), just(args_close.clone()))
                .map_with(|exprs, e| (exprs, e.span()))
                .boxed();

            let if_ = just(Token::If)
                .ignore_then(block.clone())
                .map_with(|mut exprs, e| {
                    let cond = if !exprs.0.is_empty() {
                        exprs.0.remove(0)
                    } else {
                        Expr::Nil(exprs.1)
                    };
                    let then = exprs.0;
                    let span_end = exprs.1.end;
                    Expr::If {
                        span: e.span(),
                        cond: Box::new(cond),
                        then: Expr::Block((then, exprs.1)).boxed(),
                        else_: Expr::Nil((span_end..span_end).into()).boxed(),
                    }
                })
                .boxed();

            let while_ = just(Token::While)
                .ignore_then(block.clone())
                .map_with(|mut exprs, e| {
                    let cond = if !exprs.0.is_empty() {
                        exprs.0.remove(0)
                    } else {
                        Expr::Nil(exprs.1)
                    };
                    let body = exprs.0;
                    Expr::While {
                        span: e.span(),
                        cond: Box::new(cond),
                        body: Expr::Block((body, exprs.1)).boxed(),
                    }
                })
                .boxed();

            let try_ = just(Token::Try)
                .ignore_then(block.clone())
                .then_ignore(just(Token::Catch))
                .then(block.clone())
                .map_with(|(tries, catches), e| Expr::Try {
                    span: e.span(),
                    try_: Box::new(Expr::Block(tries)),
                    err_id: Expr::Id((Token::Id("err".to_string()), SimpleSpan::default())).boxed(),
                    catch: Box::new(Expr::Block(catches)),
                })
                .labelled("try")
                .boxed();

            let return_ = just(Token::Op(":".to_string()))
                .ignore_then(inline_expr.clone())
                .map_with(|expr, e| Expr::Return(Box::new((expr, e.span()))))
                .labelled("return")
                .boxed();

            let raise_ = just(Token::Raise)
                .ignore_then(inline_expr.clone())
                .map_with(|expr, e| Expr::Raise(Box::new((expr, e.span()))))
                .labelled("raise")
                .boxed();

            choice((
                if_.clone(),
                while_.clone(),
                try_.clone(),
                inline_expr.clone(),
                return_.clone(),
                raise_.clone(),
            ))
            .labelled("statement")
        });

        let terminated_statement = statement.clone().then_ignore(just(Token::Punc(';')));

        terminated_statement
            .clone()
            .repeated()
            .collect::<Vec<_>>()
            .then(statement.clone().or_not())
            .map_with(|(mut v, last), e| {
                if let Some(expr) = last {
                    v.push(expr);
                }
                Expr::Block((v, e.span()))
            })
            .labelled("statements")
    }

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn id(&self) -> Option<String> {
        match self {
            Expr::Id(id) => Some(id.0.str().unwrap().to_string()),
            _ => None,
        }
    }

    pub fn block(self) -> Option<Vec<Self>> {
        match self {
            Expr::Block(block) => Some(block.0),
            _ => None,
        }
    }

    pub fn pair(self) -> (Self, Self) {
        match self {
            Expr::Pair { name, value } => (*name, value.0),
            _ => panic!("Expected pair"),
        }
    }

    pub fn is_delayed_arg(&self) -> bool {
        matches!(self, Expr::DelayedArg(_))
    }

    pub fn is_statement(&self) -> bool {
        matches!(self, Expr::Statement(_))
    }

    pub fn is_id(&self) -> bool {
        matches!(self, Expr::Id(_))
    }

    pub fn is_fn(&self) -> bool {
        matches!(self, Expr::Fn { .. })
    }

    pub fn is_return(&self) -> bool {
        matches!(self, Expr::Return(_))
    }

    pub fn is_raise(&self) -> bool {
        matches!(self, Expr::Raise(_))
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Expr::Nil(_))
    }

    pub fn is_if(&self) -> bool {
        matches!(self, Expr::If { .. })
    }

    pub fn is_control_statement(&self) -> bool {
        matches!(
            self,
            Expr::If { .. } | Expr::While { .. } | Expr::Try { .. }
        )
    }

    pub fn pretty_print(&self, indent: usize) -> Vec<String> {
        let indent_str = " ".repeat(indent);
        match self {
            Expr::Lit(obj) => vec![format!("{}{}", indent_str, obj.0)],
            Expr::Nil(_) => vec![format!("{}nil", indent_str)],
            Expr::DelayedArg(_) => vec![format!("{}delayed arg", indent_str)],
            Expr::Block(block) => vec![format!("{}block", indent_str)]
                .into_iter()
                .chain(
                    block
                        .0
                        .iter()
                        .flat_map(|expr| expr.pretty_print(indent + 2))
                        .collect::<Vec<_>>(),
                )
                .collect(),
            Expr::Statement(statement) => vec![format!("{}statement", indent_str)]
                .into_iter()
                .chain(statement.0.pretty_print(indent + 2))
                .collect(),
            Expr::Error(_) => vec![format!("error")],
            Expr::Unary { op, rhs, .. } => vec![format!("{}unary", indent_str)]
                .into_iter()
                .chain(op.pretty_print(indent + 2))
                .chain(rhs.pretty_print(indent + 2))
                .collect(),
            Expr::Binary { lhs, rhs, .. } => vec![format!("{}binary", indent_str)]
                .into_iter()
                .chain(lhs.pretty_print(indent + 2))
                .chain(vec![format!("{}  binary op", indent_str)])
                .chain(rhs.pretty_print(indent + 2))
                .collect(),
            Expr::Assign {
                id, indices, value, ..
            } => vec![format!("{}assign", indent_str)]
                .into_iter()
                .chain(id.pretty_print(indent + 2))
                .chain(
                    indices
                        .iter()
                        .flat_map(|index| index.pretty_print(indent + 2)),
                )
                .chain(value.pretty_print(indent + 2))
                .collect(),
            Expr::Id(_) => vec![format!("{}id", indent_str)],
            Expr::Call { f, args, .. } => {
                if args.0.is_empty() {
                    vec![format!("{}call", indent_str)]
                        .into_iter()
                        .chain(f.pretty_print(indent + 2))
                        .chain(vec![format!("{}  (0 arg)", indent_str)])
                        .collect()
                } else {
                    vec![format!("{}call", indent_str)]
                        .into_iter()
                        .chain(f.pretty_print(indent + 2))
                        .chain(
                            args.0
                                .iter()
                                .flat_map(|arg| arg.pretty_print(indent + 2))
                                .collect::<Vec<_>>(),
                        )
                        .collect()
                }
            }
            Expr::If {
                cond, then, else_, ..
            } => vec![format!("{}if", indent_str)]
                .into_iter()
                .chain(cond.pretty_print(indent + 2))
                .chain(then.pretty_print(indent + 2))
                .chain(else_.pretty_print(indent + 2))
                .collect(),
            Expr::While { cond, body, .. } => vec![format!("{}while", indent_str)]
                .into_iter()
                .chain(cond.pretty_print(indent + 2))
                .chain(body.pretty_print(indent + 2))
                .collect(),
            Expr::IfElse(if_else) => vec![format!("{}if else", indent_str)]
                .into_iter()
                .chain(
                    if_else
                        .0
                        .iter()
                        .flat_map(|node| node.pretty_print(indent + 2)),
                )
                .collect(),
            Expr::Try {
                try_,
                err_id,
                catch,
                ..
            } => vec![format!("{}try", indent_str)]
                .into_iter()
                .chain(try_.pretty_print(indent + 2))
                .chain(err_id.pretty_print(indent + 2))
                .chain(catch.pretty_print(indent + 2))
                .collect(),
            Expr::Return(_) => vec![format!("{}return", indent_str)],
            Expr::Raise(_) => vec![format!("{}raise", indent_str)],
            Expr::Bracket(expr) => vec![format!("{}bracket", indent_str)]
                .into_iter()
                .chain(expr.0.pretty_print(indent + 2))
                .collect(),
            Expr::DataFrame(expr) => vec![format!("{}table", indent_str)]
                .into_iter()
                .chain(expr.0.iter().flat_map(|node| node.pretty_print(indent + 2)))
                .collect(),
            Expr::Matrix(expr) => vec![format!("{}matrix", indent_str)]
                .into_iter()
                .chain(expr.0.iter().flat_map(|node| node.pretty_print(indent + 2)))
                .collect(),
            Expr::Dict(dict) => vec![format!("{}dict", indent_str)]
                .into_iter()
                .chain(
                    dict.0
                        .iter()
                        .flat_map(|node| node.pretty_print(indent + 2))
                        .collect::<Vec<_>>(),
                )
                .collect(),
            Expr::List(list) => vec![format!("{}list", indent_str)]
                .into_iter()
                .chain(
                    list.0
                        .iter()
                        .flat_map(|node| node.pretty_print(indent + 2))
                        .collect::<Vec<_>>(),
                )
                .collect(),
            Expr::Pair {
                name: id,
                value: column,
            } => vec![format!("{}column", indent_str)]
                .into_iter()
                .chain(id.pretty_print(indent + 2))
                .chain(column.0.pretty_print(indent + 2))
                .collect(),
            Expr::Query {
                cmd: method,
                op,
                by,
                from,
                where_,
                limit,
                ..
            } => vec![format!("{}{} query", indent_str, method)]
                .into_iter()
                .chain(if op.is_empty() {
                    vec![]
                } else {
                    vec![format!("{}  op", indent_str)]
                        .into_iter()
                        .chain(op.iter().flat_map(|node| node.pretty_print(indent + 4)))
                        .collect()
                })
                .chain(if by.is_empty() {
                    vec![]
                } else {
                    vec![format!("{}  by", indent_str)]
                        .into_iter()
                        .chain(by.iter().flat_map(|node| node.pretty_print(indent + 4)))
                        .collect()
                })
                .chain(
                    vec![format!("{}  from", indent_str)]
                        .into_iter()
                        .chain(from.pretty_print(indent + 4)),
                )
                .chain(if where_.is_empty() {
                    vec![]
                } else {
                    vec![format!("{}  where", indent_str)]
                        .into_iter()
                        .chain(where_.iter().flat_map(|node| node.pretty_print(indent + 4)))
                        .collect()
                })
                .chain(if let Some(limit) = limit.as_ref() {
                    vec![format!("{}  limit", indent_str)]
                        .into_iter()
                        .chain(limit.pretty_print(indent + 4))
                        .collect()
                } else {
                    vec![]
                })
                .collect(),
            Expr::Fn { params, body, .. } => vec![format!("{}fn", indent_str)]
                .into_iter()
                .chain(
                    params
                        .iter()
                        .flat_map(|param| param.pretty_print(indent + 2))
                        .collect::<Vec<_>>(),
                )
                .chain(body.pretty_print(indent + 2))
                .collect(),
        }
    }
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Id(id) => id.1,
            Expr::Block(block) => block.1,
            Expr::Statement(statement) => statement.1,
            Expr::Error(error) => *error,
            Expr::Unary { span, .. } => *span,
            Expr::Binary { span, .. } => *span,
            Expr::Assign { span, .. } => *span,
            Expr::Call { span, .. } => *span,
            Expr::If { span, .. } => *span,
            Expr::While { span, .. } => *span,
            Expr::IfElse(if_else) => if_else.1,
            Expr::Try { span, .. } => *span,
            Expr::Return(return_) => return_.1,
            Expr::Raise(raise) => raise.1,
            Expr::Bracket(bracket) => bracket.1,
            Expr::DataFrame(data_frame) => data_frame.1,
            Expr::Matrix(matrix) => matrix.1,
            Expr::Dict(dict) => dict.1,
            Expr::List(list) => list.1,
            Expr::Pair { value, .. } => value.1,
            Expr::Query { span, .. } => *span,
            Expr::Fn { span, .. } => *span,
            Expr::Lit(lit) => lit.1,
            Expr::Nil(span) => *span,
            Expr::DelayedArg(span) => *span,
        }
    }

    pub fn span_end(&self) -> usize {
        match self {
            Expr::Id(id) => id.1.end,
            Expr::Block(block) => block.1.end,
            Expr::Statement(statement) => statement.1.end,
            Expr::Error(error) => error.end,
            Expr::Unary { span, .. } => span.end,
            Expr::Binary { span, .. } => span.end,
            Expr::Assign { span, .. } => span.end,
            Expr::Call { span, .. } => span.end,
            Expr::If { span, .. } => span.end,
            Expr::While { span, .. } => span.end,
            Expr::IfElse(if_else) => if_else.1.end,
            Expr::Try { span, .. } => span.end,
            Expr::Return(return_) => return_.1.end,
            Expr::Raise(raise) => raise.1.end,
            Expr::Bracket(bracket) => bracket.1.end,
            Expr::DataFrame(data_frame) => data_frame.1.end,
            Expr::Matrix(matrix) => matrix.1.end,
            Expr::Dict(dict) => dict.1.end,
            Expr::List(list) => list.1.end,
            Expr::Pair { value, .. } => value.1.end,
            Expr::Query { span, .. } => span.end,
            Expr::Fn { span, .. } => span.end,
            Expr::Lit(lit) => lit.1.end,
            Expr::Nil(span) => span.end,
            Expr::DelayedArg(span) => span.end,
        }
    }

    pub fn span_start(&self) -> usize {
        match self {
            Expr::Id(id) => id.1.start,
            Expr::Block(block) => block.1.start,
            Expr::Statement(statement) => statement.1.start,
            Expr::Error(error) => error.start,
            Expr::Unary { span, .. } => span.start,
            Expr::Binary { span, .. } => span.start,
            Expr::Assign { span, .. } => span.start,
            Expr::Call { span, .. } => span.start,
            Expr::If { span, .. } => span.start,
            Expr::While { span, .. } => span.start,
            Expr::IfElse(if_else) => if_else.1.start,
            Expr::Try { span, .. } => span.start,
            Expr::Return(return_) => return_.1.start,
            Expr::Raise(raise) => raise.1.start,
            Expr::Bracket(bracket) => bracket.1.start,
            Expr::DataFrame(data_frame) => data_frame.1.start,
            Expr::Matrix(matrix) => matrix.1.start,
            Expr::Dict(dict) => dict.1.start,
            Expr::List(list) => list.1.start,
            Expr::Pair { name, .. } => name.span_start(),
            Expr::Query { span, .. } => span.start,
            Expr::Fn { span, .. } => span.start,
            Expr::Lit(lit) => lit.1.start,
            Expr::Nil(span) => span.start,
            Expr::DelayedArg(span) => span.start,
        }
    }
}
