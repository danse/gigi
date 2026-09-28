# Internal REST API

The API runs at `https://api.internal` and speaks JSON over HTTPS. Every
request must carry a bearer token obtained from the identity service; tokens
expire after one hour and are scoped to a single service.

Endpoints:

- `GET /v1/users` — list users (paginated, `limit` and `cursor` params)
- `POST /v1/users` — create a user
- `GET /v1/usage` — per-service usage counters

Rate limits are per token: 1000 requests per minute for read endpoints, 100 for
writes. On 429 the response includes a `Retry-After` header. Fields are
versioned; breaking changes bump the `/v1/` prefix.