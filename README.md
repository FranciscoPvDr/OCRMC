# INE OCR API

Aplicación web/API para verificar y extraer datos de una INE usando OCR con Tesseract.

## Endpoints

### `GET /health`

Devuelve el estado del servicio.

### `POST /api/ine/extract`

Recibe una imagen en `multipart/form-data` con el campo `file`.

Formatos soportados:

- JPG
- PNG
- WEBP
- BMP
- TIFF
- PDF

Respuesta principal:

```json
{
  "ok": true,
  "document_type": "INE",
  "confidence": 0.75,
  "extracted": {
    "nombre": "JUAN",
    "primer_apellido": "PEREZ",
    "segundo_apellido": "LOPEZ",
    "curp": "PELJ900101HDFRPN09",
    "clave_elector": "PRLPNJ90010109H300",
    "ocr": "1234567890123",
    "cic": "123456789",
    "seccion": "1234",
    "vigencia": "2024-2034",
    "domicilio": "..."
  },
  "validation": {},
  "warnings": [],
  "raw_text": "..."
}
```

## Ejecutar localmente

Instala Tesseract OCR en tu equipo y después ejecuta:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abre:

```text
http://localhost:8000
```

## Variables de entorno

- `CORS_ORIGINS`: dominios permitidos separados por coma. Usa `*` para pruebas.
- `API_KEY`: opcional. Llave fija adicional para consumir la API.
- `ADMIN_API_KEY`: llave maestra para crear/listar/revocar API keys.
- `API_KEYS_FILE`: archivo donde se guardan hashes de API keys generadas. Por defecto `data/api_keys.json`.
- `MAX_UPLOAD_MB`: tamaño máximo del archivo en MB.
- `MAX_PDF_PAGES`: máximo de páginas PDF a procesar. Por defecto `1`.

## Generar API keys

Primero configura `ADMIN_API_KEY` en Render.

Crear una API key:

```bash
curl -X POST "https://tu-servicio.onrender.com/api/admin/api-keys" \
  -H "content-type: application/json" \
  -H "x-admin-api-key: TU_ADMIN_API_KEY" \
  -d "{\"name\":\"cloudflare-worker\"}"
```

La respuesta incluye `api_key` una sola vez. Guárdala en Cloudflare como secreto.

Listar API keys:

```bash
curl "https://tu-servicio.onrender.com/api/admin/api-keys" \
  -H "x-admin-api-key: TU_ADMIN_API_KEY"
```

Revocar una API key:

```bash
curl -X DELETE "https://tu-servicio.onrender.com/api/admin/api-keys/ID_DE_LA_KEY" \
  -H "x-admin-api-key: TU_ADMIN_API_KEY"
```

Usar una API key generada:

```bash
curl -X POST "https://tu-servicio.onrender.com/api/ine/extract" \
  -H "x-api-key: TU_API_KEY_GENERADA" \
  -F "file=@ine.pdf"
```

## Despliegue en Render con Git

1. Sube esta carpeta a un repositorio de GitHub/GitLab.
2. En Render crea un nuevo servicio tipo `Web Service`.
3. Conecta el repositorio.
4. Render detectará `render.yaml` y usará Docker.
5. Configura variables de entorno si lo necesitas:
   - `API_KEY`
   - `ADMIN_API_KEY`
   - `CORS_ORIGINS`
   - `MAX_UPLOAD_MB`
   - `MAX_PDF_PAGES`
6. Despliega.

## Despliegue en Railway con Git

Railway suele funcionar mejor para esta API porque corre Docker y puede dar mejor respuesta para procesos OCR.

1. Entra a Railway:
   - `https://railway.app`
2. Crea un nuevo proyecto:
   - `New Project`
   - `Deploy from GitHub repo`
3. Selecciona el repositorio:
   - `FranciscoPvDr/OCRMC`
4. Railway detectará:
   - `railway.json`
   - `Dockerfile`
5. Configura las variables de entorno:
   - `CORS_ORIGINS=*`
   - `MAX_UPLOAD_MB=8`
   - `MAX_PDF_PAGES=1`
   - `ADMIN_API_KEY=tu_llave_admin_secreta`
   - `API_KEY=opcional_si_quieres_llave_fija`
   - `OCR_SPACE_API_KEY=opcional_para_mejorar_pdf_o_imagen_dificil`
   - `GROQ_API_KEY=opcional_para_limpiar_y_estructurar_texto_ocr`
   - `GROQ_MODEL=llama-3.1-8b-instant`
6. Espera el build/deploy.
7. Genera un dominio público desde:
   - `Settings`
   - `Networking`
   - `Generate Domain`

La URL quedará parecida a:

```text
https://ocrmc-production.up.railway.app
```

Endpoint:

```text
https://ocrmc-production.up.railway.app/api/ine/extract
```

## Ejemplo desde Cloudflare Worker

```js
export default {
  async fetch(request, env) {
    if (request.method !== 'POST') {
      return new Response('Method Not Allowed', { status: 405 });
    }

    const formData = await request.formData();
    const file = formData.get('file');

    if (!file) {
      return Response.json({ ok: false, detail: 'Falta el archivo file' }, { status: 400 });
    }

    const proxyForm = new FormData();
    proxyForm.append('file', file);

    const response = await fetch(`${env.INE_OCR_API_URL}/api/ine/extract`, {
      method: 'POST',
      headers: env.INE_OCR_API_KEY ? { 'x-api-key': env.INE_OCR_API_KEY } : {},
      body: proxyForm,
    });

    return new Response(await response.text(), {
      status: response.status,
      headers: {
        'content-type': response.headers.get('content-type') || 'application/json',
        'access-control-allow-origin': '*',
      },
    });
  },
};
```

Configura en Cloudflare Worker:

- `INE_OCR_API_URL`: URL de Render, por ejemplo `https://tu-servicio.onrender.com`
- `INE_OCR_API_KEY`: la misma llave configurada en Render, si aplica.

## Notas importantes

Este servicio hace validación heurística por OCR. Para validación oficial de identidad se requiere integración con un proveedor autorizado o procesos KYC adicionales.
