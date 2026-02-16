# can we replace linux tools with rust?

https://github.com/uutils/coreutils
but pulls in 600 crates

# is there an effort to create libraries in rust that minimize dependencies


yes:

sudo-rs: A rewrite of the sudo command in Rust that initially accrued 135 transitive dependencies but was manually reduced to just three (libc, log, and glob) to ensure it could be fully audited.
ntpd-rs: A similar effort to rewrite the Network Time Protocol daemon with a focus on a minimal, highly audited dependency graph.


# Tooling for Dependency Auditing
The ecosystem provides several tools to help developers identify and prune unnecessary dependencies:

cargo-tree: Used to visualize the dependency graph and identify where bloat or duplicate versions are introduced.

cargo-bloat: Analyzes what takes up the most space in your executable, often pointing to specific dependency-heavy crates.

cargo-deny: Allows teams to enforce policies against specific licenses, unmaintained crates, or excessive dependency counts.

cargo-vet: Helps organizations ensure that all their dependencies have been manually reviewed by a trusted auditor.



# Best Practices for Maintenance
Maintainers are increasingly encouraged to:

Use Feature Flags: Allow users to opt-in only to the parts of a library they need, which often strips away secondary dependencies.

Avoid Default Features: Developers often use --no-default-features when adding crates to keep the initial footprint small.


use https://doc.rust-lang.org/clippy/ for extra checks in code