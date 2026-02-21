use crate::assert_eq_pepper_expr;

#[test]
fn test_expr() {
    assert_eq_pepper_expr(
        "f[;1;2];",
        vec![
            "block",
            "  call",
            "    id",
            "    delayed arg",
            "    Int'1'",
            "    Int'2'",
        ],
    );
    assert_eq_pepper_expr("f[];", vec!["block", "  call", "    id", "    (0 arg)"]);
    assert_eq_pepper_expr(
        "f[;;];",
        vec![
            "block",
            "  call",
            "    id",
            "    delayed arg",
            "    delayed arg",
            "    delayed arg",
        ],
    );
    assert_eq_pepper_expr(
        "2+3*4+5",
        vec![
            "block",
            "  binary",
            "    Int'2'",
            "    binary op",
            "    binary",
            "      Int'3'",
            "      binary op",
            "      binary",
            "        Int'4'",
            "        binary op",
            "        Int'5'",
        ],
    );

    assert_eq_pepper_expr(
        "d[`k]:v;",
        vec!["block", "  assign", "    id", "    Symbol'`k'", "    id"],
    );

    assert_eq_pepper_expr(
        "[[1 2 3; 4 5 6; 7 8 9]];",
        vec![
            "block",
            "  matrix",
            "    Int'1 2 3'",
            "    Int'4 5 6'",
            "    Int'7 8 9'",
        ],
    );

    assert_eq_pepper_expr(
        "
        if [cond;
            (1;2;3)
        ];
        if [cond;
            {a:1; b:2};
        ];
        if [cond;
            ([]sym:`a`b`c)
        ];
        ",
        vec![
            "block",
            "  if",
            "    id",
            "    block",
            "      list",
            "        Int'1'",
            "        Int'2'",
            "        Int'3'",
            "    nil",
            "  if",
            "    id",
            "    block",
            "      dict",
            "        column",
            "          id",
            "          Int'1'",
            "        column",
            "          id",
            "          Int'2'",
            "      nil",
            "    nil",
            "  if",
            "    id",
            "    block",
            "      table",
            "        column",
            "          id",
            "          Symbol'`a`b`c'",
            "    nil",
        ],
    );

    assert_eq_pepper_expr(
        "
        while [i < 100;
            (1;2;3);
            i:i+1;
        ]
        ",
        vec![
            "block",
            "  while",
            "    binary",
            "      id",
            "      binary op",
            "      Int'100'",
            "    block",
            "      list",
            "        Int'1'",
            "        Int'2'",
            "        Int'3'",
            "      assign",
            "        id",
            "        binary",
            "          id",
            "          binary op",
            "          Int'1'",
            "      nil",
        ],
    );

    assert_eq_pepper_expr(
        "
        try [
            `a * 5;
        ]catch[
            err like \"*not allowed\"
        ]
        ",
        vec![
            "block",
            "  try",
            "    block",
            "      binary",
            "        Symbol'`a'",
            "        binary op",
            "        Int'5'",
            "      nil",
            "    id",
            "    block",
            "      binary",
            "        id",
            "        binary op",
            "        Str'*not allowed'",
        ],
    );

    assert_eq_pepper_expr(
        "
        f: {[a;b;c]
            $ [ a > 0; : b; c > 0; 1; 2];
            raise c;
            if [a + c = 5;
                : b;
            ]
        }
        ",
        vec![
            "block",
            "  assign",
            "    id",
            "    fn",
            "      id",
            "      id",
            "      id",
            "      block",
            "        if else",
            "          binary",
            "            id",
            "            binary op",
            "            Int'0'",
            "          return",
            "          binary",
            "            id",
            "            binary op",
            "            Int'0'",
            "          Int'1'",
            "          Int'2'",
            "        raise",
            "        if",
            "          binary",
            "            id",
            "            binary op",
            "            binary",
            "              id",
            "              binary op",
            "              Int'5'",
            "          block",
            "            return",
            "            nil",
            "          nil",
        ],
    )
}

#[test]
fn test_query_expr() {
    assert_eq_pepper_expr(
        "select from trade;",
        vec!["block", "  Select query", "    from", "      id"],
    );

    assert_eq_pepper_expr(
        "select ric: sym, time: time + 1, qty+5, price from trade limit 100",
        vec![
            "block",
            "  Select query",
            "    op",
            "      column",
            "        id",
            "        id",
            "      column",
            "        id",
            "        binary",
            "          id",
            "          binary op",
            "          Int'1'",
            "      binary",
            "        id",
            "        binary op",
            "        Int'5'",
            "      id",
            "    from",
            "      id",
            "    limit",
            "      Int'100'",
        ],
    );

    assert_eq_pepper_expr(
        "select by 0D00:00:01 xbar time from trade;",
        vec![
            "block",
            "  Select query",
            "    by",
            "      binary",
            "        Duration'0D00:00:01'",
            "        binary op",
            "        id",
            "    from",
            "      id",
        ],
    );

    assert_eq_pepper_expr(
        "select by 0D00:00:01 xbar time from trade where sym=`AAPL;",
        vec![
            "block",
            "  Select query",
            "    by",
            "      binary",
            "        Duration'0D00:00:01'",
            "        binary op",
            "        id",
            "    from",
            "      id",
            "    where",
            "      binary",
            "        id",
            "        binary op",
            "        Symbol'`AAPL'",
        ],
    );

    assert_eq_pepper_expr(
        "delete from t where c=5, d=6;",
        vec![
            "block",
            "  Delete query",
            "    from",
            "      id",
            "    where",
            "      binary",
            "        id",
            "        binary op",
            "        Int'5'",
            "      binary",
            "        id",
            "        binary op",
            "        Int'6'",
        ],
    );

    assert_eq_pepper_expr(
        "delete a, b, from t;",
        vec![
            "block",
            "  Delete query",
            "    op",
            "      id",
            "      id",
            "    from",
            "      id",
        ],
    );

    assert_eq_pepper_expr(
        "delete from t;",
        vec!["block", "  Delete query", "    from", "      id"],
    );
}

#[test]
fn test_comment_expr() {
    assert_eq_pepper_expr(
        "
        // comment
        f[ a; b;] // comment
        g[c]; // comment
        if[cond;
          // comment
          :1
        ]
        ",
        vec![
            "block",
            "  unary",
            "    call",
            "      id",
            "      id",
            "      id",
            "      delayed arg",
            "    call",
            "      id",
            "      id",
            "  if",
            "    id",
            "    block",
            "      return",
            "    nil",
        ],
    );

    assert_eq_pepper_expr(
        "
        f[
          a; // comment
          b;
        ]
        ",
        vec![
            "block",
            "  call",
            "    id",
            "    id",
            "    id",
            "    delayed arg",
        ],
    );

    assert_eq_pepper_expr(
        "delete a, // comment
          // comment
          b, // comment
          from t;",
        vec![
            "block",
            "  Delete query",
            "    op",
            "      id",
            "      id",
            "    from",
            "      id",
        ],
    );

    assert_eq_pepper_expr(
        "1 + /* comment */ 2",
        vec![
            "block",
            "  binary",
            "    Int'1'",
            "    binary op",
            "    Int'2'",
        ],
    );

    assert_eq_pepper_expr(
        "`f32$last price",
        vec![
            "block",
            "  binary",
            "    Symbol'`f32'",
            "    binary op",
            "    unary",
            "      id",
            "      id",
        ],
    );

    assert_eq_pepper_expr(
        "f1 count df",
        vec![
            "block",
            "  unary",
            "    id",
            "    unary",
            "      id",
            "      id",
        ],
    );
}
