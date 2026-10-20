# NEXT_STEP

## Hecho

- **2026-08-17** — Corregido el número de países de **63 a 62** en toda la web publicada
  (https://upocuantitativo.github.io/iafordevelopment/). Commit `2e00c65`.
  - `index.html` (tarjeta "Analysis Summary" + "Analysis covers ... countries"),
    `resultados/index.html` (las dos mismas), `panel_central.html`, `README.md`.
  - `resultados/dashboards/index.html` y `model_performance_summary.html` ya decían 62.
- **2026-08-17** — Restyle académico de `index.html` (SOLO el HTML del sitio; ninguna figura tocada).
  - Hoja de estilos reescrita conservando las 42 clases en uso (verificado: ninguna quedó sin regla).
  - Tipografía serif (Georgia) para texto y titulares; sans en small caps para etiquetas y botones.
  - Paleta institucional: tinta `#16283f` + granate `#7b2426`; fuera degradados morado/cian,
    sombras y esquinas redondeadas. Fichas con filete de 1px y regla superior.
  - Eliminados los emojis decorativos de los 34 titulares (h1-h4) y de las 3 pestañas.
  - Sustituidos los colores heredados incrustados en `style="..."` (7 casos: `#4facfe`,
    degradado `#667eea/#764ba2`, degradado `#28a745/#20c997`) por los tokens nuevos.
  - Verificado en navegador a 1400px en las 3 pestañas.

## Siguiente paso

**Bug funcional preexistente (NO introducido por el restyle):** la consola lanza
`ReferenceError: countryData is not defined` en `index.html:1801` (`updateCountrySelector`),
y por eso el desplegable "Select Country" de la pestaña 2 sale vacío, solo con
"-- Custom Values --". Comprobado que el error es idéntico en la versión anterior al restyle
(`git show HEAD~1:index.html`). Hay que revisar por qué `const countryData` (línea ~1514) no
está disponible cuando se llama al selector.

**Pendientes opcionales de estilo:**
- `panel_central.html` y `resultados/index.html` siguen con el estilo antiguo (degradado morado,
  cian, emojis). Si se quieren alineados con el nuevo, aplicar la misma hoja.
- Quedan emojis en las etiquetas de los sliders (`.control-icon`) y en las recomendaciones que
  genera el JS. Se dejaron a propósito: son iconos funcionales, no decoración de titular.

- **2026-10-20** — Añadida la carpeta `robustness/` (Code S1 de la revisión R1 de socsci-4597945 y sus resultados) y
  corregido `resultados/reportes/executive_summary.txt`, que aún era la versión de 63 países con la India y R² de una sola partición.
