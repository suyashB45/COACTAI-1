# Environment Setup Guide

This guide will help you configure the environment variables needed to run the CoAct.AI application.

## Quick Start

1. **Copy the example environment files**:
   ```powershell
   # Root directory
   Copy-Item .env.example .env
   
   # Backend directory
   Copy-Item inter-ai-backend\.env.example inter-ai-backend\.env
   ```

2. **Update the `.env` files** with your actual credentials (see sections below)

3. **Verify the configuration** by running the health check endpoint

## Required Services

### Azure OpenAI

You'll need an Azure OpenAI resource with the following deployments:

- **GPT Model**: `gpt-4.1-mini` (or your preferred model)
- **Text-to-Speech**: `tts`
- **Speech-to-Text**: `whisper`
- **Embeddings**: `text-embedding-ada-002`

**How to obtain credentials**:
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Go to **Keys and Endpoint**
4. Copy `KEY 1` and the `Endpoint` URL

**Update in `.env`**:
```env
AZURE_OPENAI_API_KEY=<your-key-here>
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
```

### Azure Blob Storage

Used for storing audio recordings and practice session data.

**How to obtain credentials**:
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Storage Account
3. Go to **Access Keys**
4. Copy the **Connection String**

**Update in `.env`**:
```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=<account>;AccountKey=<key>;EndpointSuffix=core.windows.net
```

### Supabase (PostgreSQL Database)

CoAct.AI uses Supabase for user authentication and data persistence.

**How to obtain credentials**:
1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Go to **Settings** > **Database**
4. Copy the **Connection String** (make sure to replace `[YOUR-PASSWORD]` with your actual password)
5. Go to **Settings** > **API**
6. Copy the `URL`, `anon/public` key, and `service_role` key

**Update in `.env`**:
```env
DATABASE_URL=postgresql://postgres:<password>@<project-ref>.supabase.co:6543/postgres?sslmode=require
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<anon-key>
SUPABASE_SERVICE_KEY=<service-role-key>
```

> **⚠️ Warning**: The `SUPABASE_SERVICE_KEY` bypasses Row Level Security. Keep it secret and only use it in backend code.

### CORS Configuration

Update the `CORS_ORIGINS` to include all domains that will access your backend:

```env
CORS_ORIGINS=http://localhost,http://localhost:3000,https://your-production-domain.com
```

## Configuration Files

### Root `.env`
Contains deployment-wide configuration. Used by Docker Compose and deployment scripts.

### Backend `.env` (`inter-ai-backend/.env`)
Contains backend-specific configuration. Used by the Flask application.

> **📝 Note**: Some variables are duplicated between root and backend `.env` files to support both Docker and local development workflows.

## Verification

### Test Backend Connection

```powershell
cd inter-ai-backend
python app.py
```

Visit `http://localhost:5001/health` (or your configured port) to verify the backend is running.

### Test Azure OpenAI Connection

```powershell
# In the backend directory
python -c "import os; from openai import AzureOpenAI; client = AzureOpenAI(api_key=os.getenv('AZURE_OPENAI_API_KEY'), api_version='2024-12-01-preview', azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')); print('✅ Azure OpenAI connection successful')"
```

### Test Database Connection

```powershell
# Test with psql (if installed)
# Extract connection details from DATABASE_URL and connect
```

## Security Best Practices

### 🔒 Never Commit `.env` Files

The `.gitignore` is configured to exclude all `.env` files. Always verify before committing:

```powershell
git status
# Ensure .env files are not listed
```

### 🔄 Rotate Credentials Regularly

- Change API keys every 90 days
- Use Azure Key Vault for production secrets
- Enable Azure AD authentication where possible

### 🌍 Environment-Specific Configurations

For production deployments:
- Use separate Azure resources (OpenAI, Storage, Database) for dev/staging/prod
- Configure environment variables via deployment platform (Azure App Service, Docker secrets, etc.)
- Never store production credentials in local `.env` files

### 🔍 Audit Access

Regularly review:
- Who has access to Azure Portal
- Supabase project members
- Service principal permissions

## Troubleshooting

### `DeploymentNotFound` Error

**Symptom**: Error when calling Azure OpenAI API mentioning deployment not found.

**Solution**:
1. Verify the endpoint URL is correct (should end with `.openai.azure.com/`)
2. Check deployment names match exactly in Azure Portal and `.env`
3. Ensure API version is compatible: `2024-12-01-preview`

### Database Connection Issues

**Symptom**: `psycopg2.OperationalError` or connection timeout.

**Solution**:
1. Verify the `DATABASE_URL` password is URL-encoded (e.g., `@` becomes `%40`)
2. Check Supabase project is not paused
3. Verify IP restrictions in Supabase dashboard
4. Ensure `sslmode=require` is included

### CORS Errors

**Symptom**: Browser console shows CORS policy errors.

**Solution**:
1. Add frontend domain to `CORS_ORIGINS`
2. Ensure no trailing slashes in domain names
3. Include protocol (`http://` or `https://`)

## Getting Help

- **Azure OpenAI**: [Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- **Supabase**: [Documentation](https://supabase.com/docs)
- **Project Issues**: Create an issue in the repository

## Development vs Production

### Development Setup
```env
FLASK_ENV=development
CORS_ORIGINS=http://localhost,http://localhost:3000
```

### Production Setup
```env
FLASK_ENV=production
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
```

---

**Last Updated**: January 2026
