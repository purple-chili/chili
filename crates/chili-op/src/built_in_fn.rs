use crate::{
    basic, collection, df, io, math, matrix,
    operator::{self, TRUE_DIV_OP},
    str, sys, temporal,
};
use chili_core::Func;
use std::{collections::HashMap, sync::LazyLock};

pub static BUILT_IN_FN: LazyLock<HashMap<String, Func>> = LazyLock::new(|| {
    [
        (
            "!=".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::ne)), 2, "!=", &["p1", "p2"]),
        ),
        (
            "<=".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::lt_eq)), 2, "<=", &["p1", "p2"]),
        ),
        (
            ">=".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::gt_eq)), 2, ">=", &["p1", "p2"]),
        ),
        (
            ">".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::gt)), 2, ">", &["p1", "p2"]),
        ),
        (
            "<".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::lt)), 2, "<", &["p1", "p2"]),
        ),
        (
            "=".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::eq)), 2, "=", &["p1", "p2"]),
        ),
        (
            "~".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::match_op)), 2, "~", &["p1", "p2"]),
        ),
        (
            "@".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(chili_core::at)),
                2,
                "@",
                &["collection", "indices"],
            ),
        ),
        (
            ".".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(operator::apply)),
                2,
                ".",
                &["collection", "indices"],
            ),
        ),
        (
            "$".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(operator::cast)),
                2,
                "$",
                &["type_name", "args"],
            ),
        ),
        (
            "?".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(operator::rand)),
                2,
                "?",
                &["integer", "collection"],
            ),
        ),
        (
            "!".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::dict)), 2, "!", &["keys", "values"]),
        ),
        (
            "+".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::add)), 2, "+", &["p1", "p2"]),
        ),
        (
            "-".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::minus)), 2, "-", &["p1", "p2"]),
        ),
        (
            "*".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::mul)), 2, "*", &["p1", "p2"]),
        ),
        (
            TRUE_DIV_OP.to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(operator::true_div)),
                2,
                TRUE_DIV_OP,
                &["p1", "p2"],
            ),
        ),
        (
            "div".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::div)), 2, "div", &["p1", "p2"]),
        ),
        (
            "|".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::or)), 2, "|", &["p1", "p2"]),
        ),
        (
            "&".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::and)), 2, "&", &["p1", "p2"]),
        ),
        (
            "#".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(operator::take)),
                2,
                "#",
                &["integer", "collection"],
            ),
        ),
        (
            "^".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(operator::fill)),
                2,
                "^",
                &["value", "collection"],
            ),
        ),
        (
            "_".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::remove)), 2, "_", &["p1", "p2"]),
        ),
        (
            "++".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::append)), 2, "++", &["p1", "p2"]),
        ),
        // binary
        (
            "within".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::within)),
                2,
                "within",
                &["collection", "range"],
            ),
        ),
        (
            "bottom".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::bottom)), 2, "bottom", &["k", "series"]),
        ),
        (
            "corr".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::corr)), 2, "corr", &["p1", "p2"]),
        ),
        (
            "cov0".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::cov0)), 2, "cov0", &["p1", "p2"]),
        ),
        (
            "cov1".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::cov1)), 2, "cov1", &["p1", "p2"]),
        ),
        (
            "cross".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::cross)), 2, "cross", &["p1", "p2"]),
        ),
        (
            "emean".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::ewm_mean)),
                2,
                "emean",
                &["alpha", "series"],
            ),
        ),
        (
            "estd".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::ewm_std)),
                2,
                "estd",
                &["alpha", "series"],
            ),
        ),
        (
            "evar".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::ewm_var)),
                2,
                "evar",
                &["alpha", "series"],
            ),
        ),
        (
            "in".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::in_op)),
                2,
                "in",
                &["collection", "values"],
            ),
        ),
        (
            "intersect".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::intersect)),
                2,
                "intersect",
                &["p1", "p2"],
            ),
        ),
        (
            "like".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::like)),
                2,
                "like",
                &["strings", "pattern"],
            ),
        ),
        (
            "log".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::log)), 2, "log", &["value", "base"]),
        ),
        (
            "match".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::matches)),
                2,
                "match",
                &["strings", "pattern"],
            ),
        ),
        (
            "mod".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::mod_op)), 2, "mod", &["p1", "p2"]),
        ),
        (
            "join".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::join)),
                2,
                "join",
                &["separator", "strings"],
            ),
        ),
        (
            "mmax".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_max)),
                2,
                "mmax",
                &["window", "series"],
            ),
        ),
        (
            "mmean".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_mean)),
                2,
                "mmean",
                &["window", "series"],
            ),
        ),
        (
            "mmedian".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_median)),
                2,
                "mmedian",
                &["window", "series"],
            ),
        ),
        (
            "mskew".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_skew)),
                2,
                "mskew",
                &["window", "series"],
            ),
        ),
        (
            "mmin".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_min)),
                2,
                "mmin",
                &["window", "series"],
            ),
        ),
        (
            "mstd0".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_std0)),
                2,
                "mstd0",
                &["window", "series"],
            ),
        ),
        (
            "mstd1".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_std1)),
                2,
                "mstd1",
                &["window", "series"],
            ),
        ),
        (
            "msum".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_sum)),
                2,
                "msum",
                &["window", "series"],
            ),
        ),
        (
            "mvar0".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_var0)),
                2,
                "mvar0",
                &["window", "series"],
            ),
        ),
        (
            "mvar1".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::rolling_var1)),
                2,
                "mvar1",
                &["window", "series"],
            ),
        ),
        (
            "pow".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::pow)), 2, "pow", &["base", "exponent"]),
        ),
        (
            "fby".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::fby)),
                2,
                "fby",
                &["collection", "group_by"],
            ),
        ),
        (
            "when".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::when)),
                3,
                "when",
                &["condition", "then", "else"],
            ),
        ),
        (
            "quantile".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::quantile)),
                2,
                "quantile",
                &["percentile", "series"],
            ),
        ),
        (
            "reshape".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::reshape)),
                2,
                "reshape",
                &["shape", "collection"],
            ),
        ),
        (
            "rotate".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::rotate)),
                2,
                "rotate",
                &["n", "collection"],
            ),
        ),
        (
            "round".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(math::round)),
                2,
                "round",
                &["n", "collection"],
            ),
        ),
        (
            "shift".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::shift)),
                2,
                "shift",
                &["n", "collection"],
            ),
        ),
        // ("slice".to_owned(), JBuiltInFn::Slice),
        (
            "split".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::split)),
                2,
                "split",
                &["separator", "string"],
            ),
        ),
        (
            "ss".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::ss)),
                2,
                "ss",
                &["collection", "search_values"],
            ),
        ),
        (
            "ssr".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::ssr)),
                2,
                "ssr",
                &["collection", "search_values"],
            ),
        ),
        (
            "top".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::top)), 2, "top", &["k", "series"]),
        ),
        (
            "wmean".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::weighted_mean)),
                2,
                "wmean",
                &["weights", "series"],
            ),
        ),
        (
            "wsum".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::weighted_sum)),
                2,
                "wsum",
                &["weights", "series"],
            ),
        ),
        (
            "differ".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::differ)), 2, "differ", &["p1", "p2"]),
        ),
        (
            "extend".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::extend)), 2, "extend", &["df1", "df2"]),
        ),
        (
            "hstack".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::hstack)), 2, "hstack", &["df1", "df2"]),
        ),
        // system
        (
            ".os.sleep".to_owned(),
            Func::new_built_in_fn(Some(Box::new(sys::sleep)), 1, ".os.sleep", &["ms"]),
        ),
        (
            ".os.glob".to_owned(),
            Func::new_built_in_fn(Some(Box::new(sys::glob)), 1, ".os.glob", &["path"]),
        ),
        (
            ".os.setenv".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(sys::setenv)),
                2,
                ".os.setenv",
                &["name", "value"],
            ),
        ),
        (
            ".os.seed".to_owned(),
            Func::new_built_in_fn(Some(Box::new(sys::seed)), 1, ".os.seed", &["seed"]),
        ),
        (
            ".os.system".to_owned(),
            Func::new_built_in_fn(Some(Box::new(sys::system)), 1, ".os.system", &["command"]),
        ),
        (
            ".os.getenv".to_owned(),
            Func::new_built_in_fn(Some(Box::new(sys::getenv)), 1, ".os.getenv", &["name"]),
        ),
        (
            ".os.mem".to_owned(),
            Func::new_built_in_fn(Some(Box::new(sys::mem)), 0, ".os.mem", &[]),
        ),
        (
            "union".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::union)), 2, "union", &["p1", "p2"]),
        ),
        (
            "vstack".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::vstack)), 2, "vstack", &["df1", "df2"]),
        ),
        (
            "xasc".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::x_asc)), 2, "xasc", &["columns", "df"]),
        ),
        (
            "xbar".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::xbar)),
                2,
                "xbar",
                &["bar_size", "series"],
            ),
        ),
        (
            "xdesc".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::x_desc)), 2, "xdesc", &["columns", "df"]),
        ),
        (
            "xreorder".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::x_reorder)),
                2,
                "xreorder",
                &["columns", "df"],
            ),
        ),
        (
            "xrename".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::x_rename)),
                2,
                "xrename",
                &["columns", "df"],
            ),
        ),
        // unary
        (
            "abs".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::abs)), 1, "abs", &["value"]),
        ),
        (
            "all".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::all)), 1, "all", &["collection"]),
        ),
        (
            "any".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::any)), 1, "any", &["collection"]),
        ),
        (
            "acos".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::arccos)), 1, "acos", &["value"]),
        ),
        (
            "acosh".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::arccosh)), 1, "acosh", &["value"]),
        ),
        (
            "asin".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::arcsin)), 1, "asin", &["value"]),
        ),
        (
            "asinh".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::arcsinh)), 1, "asinh", &["value"]),
        ),
        (
            "atan".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::arctan)), 1, "atan", &["value"]),
        ),
        (
            "atanh".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::arctanh)), 1, "atanh", &["value"]),
        ),
        (
            "asc".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::asc)), 1, "asc", &["series"]),
        ),
        (
            "bfill".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::backward_fill)),
                1,
                "bfill",
                &["series"],
            ),
        ),
        (
            "cbrt".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::cbrt)), 1, "cbrt", &["value"]),
        ),
        (
            "ceil".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::ceil)), 1, "ceil", &["value"]),
        ),
        (
            "cos".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::cos)), 1, "cos", &["value"]),
        ),
        (
            "cosh".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::cosh)), 1, "cosh", &["value"]),
        ),
        (
            "cot".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::cot)), 1, "cot", &["value"]),
        ),
        (
            "count".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::count)),
                1,
                "count",
                &["collection"],
            ),
        ),
        (
            "ccount".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::cum_count)),
                1,
                "ccount",
                &["series"],
            ),
        ),
        (
            "cmax".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::cum_max)), 1, "cmax", &["series"]),
        ),
        (
            "cmin".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::cum_min)), 1, "cmin", &["series"]),
        ),
        (
            "cprod".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::cum_prod)),
                1,
                "cprod",
                &["series"],
            ),
        ),
        (
            "csum".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::cum_sum)), 1, "csum", &["series"]),
        ),
        (
            "desc".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::desc)), 1, "desc", &["series"]),
        ),
        (
            "diff".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::diff)), 1, "diff", &["series"]),
        ),
        (
            "exp".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::exp)), 1, "exp", &["value"]),
        ),
        (
            "first".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::first)), 1, "first", &["series"]),
        ),
        (
            "flatten".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::flatten)),
                1,
                "flatten",
                &["series"],
            ),
        ),
        (
            "floor".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::floor)), 1, "floor", &["value"]),
        ),
        (
            "fill".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::forward_fill)),
                1,
                "fill",
                &["series"],
            ),
        ),
        (
            "hash".to_owned(),
            Func::new_built_in_fn(Some(Box::new(sys::nyi)), 1, "hash", &["value"]),
        ),
        (
            "interp".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::interpolate)),
                1,
                "interp",
                &["series"],
            ),
        ),
        (
            "kurtosis".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::kurtosis)),
                1,
                "kurtosis",
                &["series"],
            ),
        ),
        (
            "last".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::last)), 1, "last", &["collection"]),
        ),
        (
            "ln".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::ln)), 1, "ln", &["value"]),
        ),
        (
            "log10".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::log10)), 1, "log10", &["value"]),
        ),
        (
            "log1p".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::log1p)), 1, "log1p", &["value"]),
        ),
        (
            "lowercase".to_owned(),
            Func::new_built_in_fn(Some(Box::new(str::lowercase)), 1, "lowercase", &["string"]),
        ),
        (
            "trims".to_owned(),
            Func::new_built_in_fn(Some(Box::new(str::trim_start)), 1, "trims", &["string"]),
        ),
        (
            "pad".to_owned(),
            Func::new_built_in_fn(Some(Box::new(str::pad)), 2, "pad", &["length", "string"]),
        ),
        (
            "max".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::max)), 1, "max", &["series"]),
        ),
        (
            "mean".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::mean)), 1, "mean", &["series"]),
        ),
        (
            "median".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::median)), 1, "median", &["series"]),
        ),
        (
            "min".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::min)), 1, "min", &["series"]),
        ),
        (
            "neg".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::neg)), 1, "neg", &["value"]),
        ),
        (
            "next".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::next)), 1, "next", &["series"]),
        ),
        (
            "mode".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::mode)), 1, "mode", &["series"]),
        ),
        (
            "not".to_owned(),
            Func::new_built_in_fn(Some(Box::new(operator::not)), 1, "not", &["series"]),
        ),
        (
            "null".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::null)), 1, "null", &["series"]),
        ),
        (
            "pc".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::percent_change)),
                1,
                "pc",
                &["series"],
            ),
        ),
        (
            "prev".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::prev)), 1, "prev", &["series"]),
        ),
        (
            "prod".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::product)), 1, "prod", &["series"]),
        ),
        (
            "rank".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::rank)), 1, "rank", &["series"]),
        ),
        (
            "reverse".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::reverse)),
                1,
                "reverse",
                &["series"],
            ),
        ),
        (
            "trime".to_owned(),
            Func::new_built_in_fn(Some(Box::new(str::trim_end)), 1, "trime", &["series"]),
        ),
        (
            "shuffle".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::shuffle)),
                1,
                "shuffle",
                &["series"],
            ),
        ),
        (
            "sign".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::sign)), 1, "sign", &["series"]),
        ),
        (
            "sin".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::sin)), 1, "sin", &["value"]),
        ),
        (
            "sinh".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::sinh)), 1, "sinh", &["value"]),
        ),
        (
            "skew".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::skew)), 1, "skew", &["series"]),
        ),
        (
            "sqrt".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::sqrt)), 1, "sqrt", &["value"]),
        ),
        (
            "std0".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::std0)), 1, "std0", &["series"]),
        ),
        (
            "std1".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::std1)), 1, "std1", &["series"]),
        ),
        (
            "trim".to_owned(),
            Func::new_built_in_fn(Some(Box::new(str::trim)), 1, "trim", &["string"]),
        ),
        (
            "sum".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::sum)), 1, "sum", &["series"]),
        ),
        (
            "tan".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::tan)), 1, "tan", &["value"]),
        ),
        (
            "tanh".to_owned(),
            Func::new_built_in_fn(Some(Box::new(math::tanh)), 1, "tanh", &["value"]),
        ),
        (
            "unique".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::unique)), 1, "unique", &["series"]),
        ),
        (
            "uc".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::unique_count)),
                1,
                "uc",
                &["series"],
            ),
        ),
        (
            "uppercase".to_owned(),
            Func::new_built_in_fn(Some(Box::new(str::uppercase)), 1, "uppercase", &["string"]),
        ),
        (
            "var0".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::var0)), 1, "var0", &["series"]),
        ),
        (
            "var1".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::var1)), 1, "var1", &["series"]),
        ),
        (
            "cols".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::cols)), 1, "cols", &["df"]),
        ),
        (
            "describe".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::describe)), 1, "describe", &["df"]),
        ),
        (
            "enlist".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::enlist)), 1, "enlist", &["value"]),
        ),
        (
            "filter".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::filter)), 1, "filter", &["series"]),
        ),
        (
            "flag".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::flag)), 1, "flag", &["series"]),
        ),
        (
            "exists".to_owned(),
            Func::new_built_in_fn(Some(Box::new(io::exists)), 1, "exists", &["string"]),
        ),
        (
            "hdel".to_owned(),
            Func::new_built_in_fn(Some(Box::new(io::h_del)), 1, "hdel", &["string"]),
        ),
        (
            "key".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::keys)), 1, "key", &["dict"]),
        ),
        (
            "value".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::values)), 1, "value", &["dict"]),
        ),
        (
            "schema".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::schema)), 1, "schema", &["df"]),
        ),
        (
            "show".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::show)), 1, "show", &["series"]),
        ),
        (
            "range".to_owned(),
            Func::new_built_in_fn(Some(Box::new(collection::range)), 1, "range", &["series"]),
        ),
        (
            "transpose".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::transpose)),
                1,
                "transpose",
                &["df"],
            ),
        ),
        (
            "type".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::type_op)), 1, "type", &["args"]),
        ),
        // other
        (
            "aj".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::aj)), 3, "aj", &["columns", "df", "df"]),
        ),
        (
            "cj".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::cj)), 3, "cj", &["columns", "df", "df"]),
        ),
        (
            "ij".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::ij)), 3, "ij", &["columns", "df", "df"]),
        ),
        (
            "lj".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::lj)), 3, "lj", &["columns", "df", "df"]),
        ),
        (
            "fj".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::fj)), 3, "fj", &["columns", "df", "df"]),
        ),
        // (
        //     "pj".to_owned(),
        //     Func::new_built_in_fn(Some(Box::new(sys::nyi)), 3, "pj", &["columns", "df", "df"]),
        // ),
        (
            "wj".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::wj)),
                7,
                "wj",
                &[
                    "by_columns",
                    "time",
                    "start",
                    "end",
                    "aggregations",
                    "df0",
                    "df1",
                ],
            ),
        ),
        (
            "anti".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::anti)),
                3,
                "anti",
                &["columns", "df", "df"],
            ),
        ),
        (
            "semi".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::semi)),
                3,
                "semi",
                &["columns", "df", "df"],
            ),
        ),
        (
            "console".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(sys::console)),
                2,
                "console",
                &["rows", "cols"],
            ),
        ),
        (
            "assert".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::assert)), 1, "assert", &["condition"]),
        ),
        (
            "equal".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::equal)), 2, "equal", &["left", "right"]),
        ),
        (
            "fail".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(sys::nyi)),
                2,
                "fail",
                &["list", "error_msg_pattern"],
            ),
        ),
        (
            "rcsv".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(io::read_csv)),
                5,
                "rcsv",
                &["file", "has_header", "separator", "ignore_errors", "dtypes"],
            ),
        ),
        (
            "rdatabase".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(sys::nyi)),
                2,
                "rdatabase",
                &["database_url", "sql"],
            ),
        ),
        (
            "rexcel".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(sys::nyi)),
                2,
                "rexcel",
                &["path", "sheet_name"],
            ),
        ),
        (
            "rjson".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(io::read_json)),
                2,
                "rjson",
                &["path", "dtypes"],
            ),
        ),
        (
            "rparquet".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(io::read_parquet)),
                4,
                "rparquet",
                &["path", "n_rows", "rechunk", "columns"],
            ),
        ),
        (
            "rtxt".to_owned(),
            Func::new_built_in_fn(Some(Box::new(io::read_txt)), 1, "rtxt", &["path"]),
        ),
        (
            "wcsv".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(io::write_csv)),
                3,
                "wcsv",
                &["file", "df", "separator"],
            ),
        ),
        (
            "wdatabase".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(sys::nyi)),
                3,
                "wdatabase",
                &["database_url", "table_name", "df"],
            ),
        ),
        (
            "wexcel".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(sys::nyi)),
                3,
                "wexcel",
                &["path", "sheet_name", "df"],
            ),
        ),
        (
            "wjson".to_owned(),
            Func::new_built_in_fn(Some(Box::new(io::write_json)), 2, "wjson", &["path", "df"]),
        ),
        (
            "wparquet".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(io::write_parquet)),
                3,
                "wparquet",
                &["path", "df", "compress_level"],
            ),
        ),
        (
            "wpar".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(io::write_partition)),
                6,
                "wpar",
                &[
                    "path",
                    "partition",
                    "table",
                    "df",
                    "sort_columns",
                    "rechunk",
                ],
            ),
        ),
        (
            "wtxt".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(io::write_txt)),
                3,
                "wtxt",
                &["path", "strings", "append_mode"],
            ),
        ),
        (
            "inv".to_owned(),
            Func::new_built_in_fn(Some(Box::new(matrix::inv)), 1, "inv", &["matrix"]),
        ),
        // query other
        (
            "clip".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::clip)),
                3,
                "clip",
                &["series", "min", "max"],
            ),
        ),
        (
            "concat".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::concat)),
                3,
                "concat",
                &["sep", "left", "right"],
            ),
        ),
        (
            "replace".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(str::replace)),
                3,
                "replace",
                &["strings", "pattern", "replacement"],
            ),
        ),
        (
            "mquantile".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(collection::rolling_quantile)),
                3,
                "mquantile",
                &["percentile", "size", "series"],
            ),
        ),
        (
            "unpivot".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::unpivot)),
                3,
                "unpivot",
                &["df", "indices", "on_cols"],
            ),
        ),
        (
            "pivot".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::pivot)),
                5,
                "pivot",
                &["df", "indices", "on_cols", "values", "agg_fn_name"],
            ),
        ),
        (
            "lit".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::lit)), 1, "lit", &["args"]),
        ),
        (
            "col".to_owned(),
            Func::new_built_in_fn(Some(Box::new(basic::col)), 1, "col", &["string"]),
        ),
        (
            "as".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(basic::as_op)),
                2,
                "as",
                &["expr", "column_name"],
            ),
        ),
        (
            "flip".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::flip)), 1, "flip", &["dict"]),
        ),
        (
            "explode".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(df::explode)),
                2,
                "explode",
                &["df", "columns"],
            ),
        ),
        (
            "collect".to_owned(),
            Func::new_built_in_fn(Some(Box::new(df::collect)), 1, "collect", &["lf"]),
        ),
        (
            "now".to_owned(),
            Func::new_built_in_fn(Some(Box::new(temporal::now)), 1, "now", &["timezone"]),
        ),
        (
            "today".to_owned(),
            Func::new_built_in_fn(Some(Box::new(temporal::today)), 1, "today", &["timezone"]),
        ),
        (
            "utc".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(temporal::utc)),
                2,
                "utc",
                &["timestamp", "timezone"],
            ),
        ),
        (
            "local".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(temporal::local)),
                2,
                "local",
                &["timestamp", "timezone"],
            ),
        ),
        (
            "tz".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(temporal::tz)),
                3,
                "tz",
                &["timestamp", "from_timezone", "to_timezone"],
            ),
        ),
    ]
    .into_iter()
    .collect()
});
