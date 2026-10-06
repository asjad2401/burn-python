// Turning Rust panics into Python exceptions.
//
// Panics inside Burn are caught and re-raised with their message, so the default hook's
// stderr dump is just noise (RUST_BACKTRACE keeps it). GPU backends fail on a worker
// thread, and the panic we catch is only a downstream `RecvError`; the hook records the
// first panic on any thread so that root cause can be reported instead.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

static FIRST_PANIC: Mutex<Option<String>> = Mutex::new(None);

pub fn install_hook() {
    let default_hook = panic::take_hook();
    let verbose = std::env::var_os("RUST_BACKTRACE").is_some();
    panic::set_hook(Box::new(move |info| {
        if let Ok(mut first) = FIRST_PANIC.lock() {
            first.get_or_insert_with(|| payload_message(info.payload()).to_string());
        }
        if verbose {
            default_hook(info);
        }
    }));
}

fn payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic")
}

/// Run `f`, turning a panic (on this or a backend worker thread) into its message.
pub fn catch<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    if let Ok(mut first) = FIRST_PANIC.lock() {
        *first = None;
    }
    panic::catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        FIRST_PANIC
            .lock()
            .ok()
            .and_then(|mut first| first.take())
            .unwrap_or_else(|| payload_message(&*payload).to_string())
    })
}
