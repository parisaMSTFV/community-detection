# Security and publication boundary

The default project runs offline and has no credential, network, database, or dashboard dependency.
CI scans tracked and non-ignored untracked text for common credential and private-infrastructure
patterns.

External analysis pseudonymizes identifiers by default with HMAC-SHA256 and a salt supplied through
`COMMUNITY_DETECTION_ID_SALT`. Never commit that salt. Pseudonymization reduces accidental exposure
but does not make customer-level data anonymous or automatically safe to publish.

Do not add real user identifiers, category taxonomies, graph exports, queries, private URLs,
screenshots, credentials, or production configuration to this repository. Use
`--allow-raw-identifiers` only for explicitly approved public or fully synthetic inputs, and protect
all generated artifacts according to the source data policy.
