use std::io::Write;

use reedline::ExternalPrinter;

pub struct Pipe {
    printer: ExternalPrinter<String>,
}

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
        match self
            .printer
            .print(String::from_utf8(buf.to_vec()).unwrap_or_default())
        {
            Ok(_) => Ok(buf.len()),
            Err(e) => Err(std::io::Error::other(e)),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Pipe {
    pub fn new(printer: ExternalPrinter<String>) -> Self {
        Self { printer }
    }
}

unsafe impl Send for Pipe {}
