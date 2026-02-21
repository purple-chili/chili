use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::prelude::*;

pub fn print_errs<'a, T: ToString + Clone>(
    errs: Vec<Rich<'a, T>>,
    filename: &'a str,
    src: &'a str,
) {
    errs.into_iter()
        .map(|e| e.map_token(|c| c.to_string()))
        .for_each(|e| {
            Report::build(ReportKind::Error, (filename, e.span().into_range()))
                .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
                .with_message(e.to_string())
                .with_label(
                    Label::new((filename, e.span().into_range()))
                        .with_message(e.reason().to_string())
                        .with_color(Color::Red),
                )
                .with_labels(e.contexts().map(|(label, span)| {
                    Label::new((filename, span.into_range()))
                        .with_message(format!("while parsing this {label}"))
                        .with_color(Color::Yellow)
                }))
                .finish()
                .print((filename, Source::from(src)))
                .unwrap()
        })
}

pub fn get_err_msg<'a, T: ToString + Clone>(
    errs: Vec<Rich<'a, T>>,
    filename: &'a str,
    src: &'a str,
) -> String {
    errs.into_iter()
        .map(|e| e.map_token(|c| c.to_string()))
        .map(|e| {
            let report = Report::build(ReportKind::Error, (filename, e.span().into_range()))
                .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
                .with_message(e.to_string())
                .with_label(
                    Label::new((filename, e.span().into_range()))
                        .with_message(e.reason().to_string())
                        .with_color(Color::Red),
                )
                .with_labels(e.contexts().map(|(label, span)| {
                    Label::new((filename, span.into_range()))
                        .with_message(format!("while parsing this {label}"))
                        .with_color(Color::Yellow)
                }));
            let mut buf = Vec::new();
            report
                .finish()
                .write_for_stdout((filename, Source::from(src)), &mut buf)
                .unwrap();
            String::from_utf8(buf).unwrap()
        })
        .collect::<Vec<String>>()
        .join("\n")
}
