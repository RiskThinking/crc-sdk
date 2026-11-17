# Source resource terms

[Docs index](README.md) · [Canonical datasets](datasets.md)

The SDK code is licensed under [AGPL-3.0-or-later](../LICENSE). Source resource
terms are separate data recorded with each acquired resource.

A `ResourceIdentity` carries its URI and full content SHA-256 identity. When
known, its `license`, `attribution`, and timezone-aware `retrieved_at` retain the
source's actual terms and retrieval context. Unknown facts remain absent;
there is no invented default license or attribution.

Source identities name the provider, product, release, and member. Execution
reports retain contributing source identities, and typed provenance records
transformations such as unit conversion and identity changes. This keeps source
terms and the operations performed on their data inspectable together.

Metadata is evidence of what was recorded for a resource, rather than a policy
that automatically assigns the SDK's code license to source data. Publishers
remain the authority for their resource terms and citation requirements.
