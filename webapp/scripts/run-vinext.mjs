// Set the local log path without POSIX-only environment assignment syntax.
// This keeps npm dev/build/start usable from Windows, macOS and Linux.
process.env.WRANGLER_LOG_PATH ??= ".wrangler/wrangler.log";
await import(new URL("../node_modules/vinext/dist/cli.js", import.meta.url).href);
