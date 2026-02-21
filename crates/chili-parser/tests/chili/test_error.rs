use crate::should_fail_chili_expr;

#[test]
fn test_error() {
    should_fail_chili_expr("delete from t where c=5, d=6,", "last comma");

    should_fail_chili_expr("[select from t, update from t where c within 4 5, 1 2 3,, (select from t where a=1,b=2), 4 5 6]", "empty expression in the list");

    should_fail_chili_expr("d():v", "empty indices");

    should_fail_chili_expr(
        "function(xasc){}",
        "built-in functions cannot be used as parameter name",
    );
}
