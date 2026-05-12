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
- `API_KEY`: opcional. Si se configura, las solicitudes deben enviar header `x-api-key`.
- `MAX_UPLOAD_MB`: tamaño máximo del archivo en MB.

## Despliegue en Render con Git

1. Sube esta carpeta a un repositorio de GitHub/GitLab.
2. En Render crea un nuevo servicio tipo `Web Service`.
3. Conecta el repositorio.
4. Render detectará `render.yaml` y usará Docker.
5. Configura variables de entorno si lo necesitas:
   - `API_KEY`
   - `CORS_ORIGINS`
   - `MAX_UPLOAD_MB`
6. Despliega.

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
