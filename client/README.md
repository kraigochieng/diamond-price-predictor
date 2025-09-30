## Client (Nuxt 4)

Nuxt 4 single‑page app for diamond price prediction. It calls the FastAPI server to obtain a prediction from the MLflow‑managed model.

### Prerequisites

-   Node.js 18+ (20+ recommended)

### Environment

This app reads `NUXT_PUBLIC_API_BASE` from the root `.env` via Nuxt runtime config.

`nuxt.config.ts` defaults to `http://localhost:8000` if not set:

```10:16:/home/kraigochieng/projects/diamond-price-predictor/client/nuxt.config.ts
        public: {
            apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000',
        }
```

### Install

```bash
npm install
```

### Run (dev)

```bash
npm run dev
```

Opens `http://localhost:3000`.

### Build

```bash
npm run build
```

### Preview production build

```bash
npm run preview
```

### Data contract

Current minimal payload is a subset focused on `carat`:

```1:12:/home/kraigochieng/projects/diamond-price-predictor/client/app/types/diamond.ts
export interface DiamondInput {
    carat: number;
}

export interface DiamondOutput {
    message: string;
    data: {
        carat: number;
    };
    df: string;
    preds: number;
}
```

If the server expands inputs (e.g., cut/color/clarity, dimensions), update these types and the form accordingly.

### Troubleshooting

-   404/Network error: ensure the API is running and `NUXT_PUBLIC_API_BASE` points to it.
-   CORS error: ensure the server `CLIENT_URL` includes `http://localhost:3000`.
