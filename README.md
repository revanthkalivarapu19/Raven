# RAVEN Frontend

RAVEN (Real-time Agentic Verification & Evidence-based Fake News Detection) is an AI-assisted, editorial chat-style claim verification web interface built with React and Vite. It allows users to submit factual claims (as text and/or screenshots in English or Telugu) and inspect evidence-backed verdicts, confidence rings, and source credibility breakdowns.

---

## Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/) (v18 or higher recommended)
- `npm` (bundled with Node.js)

### Installation
Install the project dependencies:
```bash
# Using lockfile
npm ci

# Or standard install
npm install
```

### Environment Configuration
Copy the example environment configuration if needed:
```bash
cp .env.example .env.local
```
Key configuration settings:
- `VITE_ENABLE_SAMPLES=true`: Enables local mock/demo verification scenarios (Real, Fake, Unverified) without needing a live backend connection.
- `VITE_API_BASE_URL`: Optional custom base URL when the backend API service is hosted on a separate host or port (defaults to relative `/api`).

### Running Locally
To start the Vite development server:
```bash
npm run dev
```
The application will be available at `http://localhost:5173`.

### Running Linter
```bash
npm run lint
```

### Building for Production
To bundle the frontend application for production:
```bash
npm run build
```
The compiled output will be generated in the `dist/` directory.

### Previewing the Production Build
```bash
npm run preview
```

---

## Development & Prototype Mode

When running in disconnected/prototype mode without a connected backend server:
- **Via URL parameter:** Append `?sample=true` to the URL (e.g., `http://localhost:5173/?sample=true`).
- **Via LocalStorage:** Open browser DevTools console and execute:
  ```js
  localStorage.setItem('RAVEN_ENABLE_SAMPLES', 'true');
  ```
  *(To disable: `localStorage.removeItem('RAVEN_ENABLE_SAMPLES')`)*
- **Via Environment Variable:** Set `VITE_ENABLE_SAMPLES=true` in `.env.local`.

---

## API & Service Layer Specifications

The frontend communicates with fact-checking services via `src/services/ravenApi.js`.

### 1. `verify({ text, image })`

Evaluates a claim provided as text, an attached screenshot, or both.

#### Request Parameters:
```typescript
interface VerifyRequest {
  text?: string;                // Claim text statement (English or Telugu)
  image?: {                     // Optional attached screenshot
    name: string;               // File name
    preview: string;            // Base64 data URL
  } | null;
}
```

#### Response Shape:
```typescript
interface VerifyResponse {
  verdict: 'real' | 'fake' | 'unverified';
  confidence: number;           // Integer between 0 and 100
  status: string;               // Single concise sentence stating conclusion
  why: string;                  // Short explanation summarizing the findings
  evidence: EvidenceSource[];   // List of analyzed sources
}

interface EvidenceSource {
  source: string;               // Name of publisher / authority
  stance: 'support' | 'contradict' | 'unclear';
  relevance: number;            // 0 - 100
  trust: number;                // 0 - 100
  excerpt: string;              // Direct quote or excerpt from publication
  note: string;                 // Editorial credibility evaluation note
}
```

### 2. `followUp({ question, result })`

Answers follow-up questions within the context of an existing verified claim.

#### Request Parameters:
```typescript
interface FollowUpRequest {
  question: string;             // User follow-up query
  result: VerifyResponse;       // Parent verification result object
}
```

#### Response:
Returns a `Promise<string>` containing a plain-text editorial answer displayed under the RAVEN label.
