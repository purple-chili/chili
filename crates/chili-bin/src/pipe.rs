use std::io::Write;

use reedline::ExternalPrinter;

pub struct Pipe {
    printer: ExternalPrinter<String>,
}

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
        // The printer's channel is small and is drained only while the REPL is
        // reading a line. A blocking send from a logging thread (env_logger
        // holds its lock across this write) would stall every thread that logs
        // while a script is loading or a long evaluation runs: never block, and
        // fall back to stderr when the channel is full.
        let msg = String::from_utf8_lossy(buf).into_owned();
        if let Err(e) = self.printer.sender().try_send(msg) {
            let msg = e.into_inner();
            let _ = std::io::stderr().write_all(msg.as_bytes());
        }
        Ok(buf.len())
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
