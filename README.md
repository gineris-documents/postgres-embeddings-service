# PostgreSQL Embeddings Service

This service processes PDF documents from Google Drive, generates text embeddings, and stores them in a PostgreSQL database for vector search.

## Workflow

1. The service checks the `ai_data.document_tracking` table for documents with `status = 'pending'`
2. For each pending document:
   - If it has `previous_drive_file_id`, delete that document's data
   - Download file from Google Drive using the `drive_file_id`
   - Process PDF and generate text chunks
   - Generate embeddings for each chunk
   - Store in PostgreSQL with proper metadata
   - Update status to `'processed'` when complete

## Setup

1. Create a service account with Google Drive access
2. Download the service account credentials as `service-account.json`
3. Deploy the service to Google Cloud Run

## Environment Variables

- `PROJECT_ID`: Google Cloud project ID
- `PG_HOST`: PostgreSQL host address
- `PG_DATABASE`: PostgreSQL database name
- `PG_USER`: PostgreSQL username
- `PG_PASSWORD`: PostgreSQL password
- `SERVICE_ACCOUNT_FILE`: Path to the service account credentials file (default: `service-account.json`)

## Deployment

```bash
gcloud run deploy postgres-embeddings-service \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
