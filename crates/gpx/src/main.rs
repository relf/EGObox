//! Command-line entrypoint for the `gpx` executable.

use anyhow::Result;

fn main() -> Result<()> {
    egobox_gpx::run()
}
