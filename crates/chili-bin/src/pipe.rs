use std::io::Write;

use std::sync::mpsc::{SyncSender, TrySendError};

pub struct Pipe {
    sender: SyncSender<String>,
}

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> Result<usize, std::io::Error> {
        // The printer's channel is small and is drained only while the REPL is
        // reading a line. A blocking send from a logging thread (env_logger
        // holds its lock across this write) would stall every thread that logs
        // while a script is loading or a long evaluation runs: never block, and
        // fall back to stderr when the channel is full.
        let msg = String::from_utf8_lossy(buf).into_owned();
        if let Err(e) = self.sender.try_send(msg) {
            let (TrySendError::Full(msg) | TrySendError::Disconnected(msg)) = e;
            let _ = std::io::stderr().write_all(msg.as_bytes());
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Pipe {
    pub fn new(sender: SyncSender<String>) -> Self {
        Self { sender }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reedline::ExternalPrinter;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    #[test]
    fn logs_reach_the_printer_through_its_sender() {
        let printer = ExternalPrinter::<String>::new(1);
        Pipe::new(printer.sender()).write_all(b"hello\n").unwrap();
        assert_eq!(printer.get_line().as_deref(), Some("hello\n"));
    }

    fn assert_write_does_not_block(sender: SyncSender<String>) {
        let (done, received) = channel();
        let worker = std::thread::spawn(move || {
            Pipe::new(sender).write_all(b"fallback\n").unwrap();
            done.send(()).unwrap();
        });
        received
            .recv_timeout(Duration::from_secs(2))
            .expect("logging blocked");
        worker.join().unwrap();
    }

    #[test]
    fn full_and_disconnected_printers_do_not_block_logging() {
        let printer = ExternalPrinter::<String>::new(1);
        printer.sender().send("queued".into()).unwrap();
        assert_write_does_not_block(printer.sender());
        assert_eq!(printer.get_line().as_deref(), Some("queued"));
        let sender = printer.sender();
        drop(printer);
        assert_write_does_not_block(sender);
    }
}
