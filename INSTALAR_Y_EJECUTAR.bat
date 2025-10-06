@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

REM =====================================================================
REM SCRIPT AUTOMÁTICO PARA WINDOWS
REM Análisis de Género y Desarrollo Económico
REM =====================================================================

cls
echo.
echo ================================================================================
echo 🚀 ANÁLISIS AVANZADO DE GÉNERO Y DESARROLLO ECONÓMICO
echo ================================================================================
echo 📊 Script automático para Windows
echo 🔧 Instala dependencias y ejecuta análisis completo
echo ================================================================================
echo.

REM Verificar Python
echo 🔍 Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado
    echo.
    echo 💡 SOLUCIÓN:
    echo    1. Descarga Python desde: https://python.org
    echo    2. Durante la instalación, marca "Add Python to PATH"
    echo    3. Reinicia esta ventana de comandos
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION% encontrado

REM Verificar archivos necesarios
echo.
echo 📁 Verificando archivos del proyecto...

set ARCHIVOS_NECESARIOS=0
set TOTAL_ARCHIVOS=0

REM Verificar scripts principales
set /a TOTAL_ARCHIVOS+=1
if exist "analisis_principal.py" (
    echo ✓ analisis_principal.py
    set /a ARCHIVOS_NECESARIOS+=1
) else (
    echo ❌ analisis_principal.py (REQUERIDO)
)

set /a TOTAL_ARCHIVOS+=1
if exist "instalar_dependencias.py" (
    echo ✓ instalar_dependencias.py
    set /a ARCHIVOS_NECESARIOS+=1
) else (
    echo ❌ instalar_dependencias.py (REQUERIDO)
)

set /a TOTAL_ARCHIVOS+=1
if exist "ejecutar.py" (
    echo ✓ ejecutar.py
    set /a ARCHIVOS_NECESARIOS+=1
) else (
    echo ❌ ejecutar.py (REQUERIDO)
)

REM Verificar archivos de datos
echo.
echo 📊 Verificando archivos de datos...
set DATOS_ENCONTRADOS=0

if exist "DATA_GHAB.xlsx" (
    echo ✓ DATA_GHAB.xlsx encontrado
    set DATOS_ENCONTRADOS=1
) else (
    echo ⚪ DATA_GHAB.xlsx no encontrado
)

if exist "paste.txt" (
    echo ✓ paste.txt encontrado
    set DATOS_ENCONTRADOS=1
) else (
    echo ⚪ paste.txt no encontrado
)

REM Verificar que tengamos archivos suficientes
if %ARCHIVOS_NECESARIOS% lss %TOTAL_ARCHIVOS% (
    echo.
    echo ❌ ERROR: Faltan archivos necesarios del proyecto
    echo 💡 Asegúrate de tener todos los archivos Python en esta carpeta
    echo.
    pause
    exit /b 1
)

if %DATOS_ENCONTRADOS% equ 0 (
    echo.
    echo ⚠️ ADVERTENCIA: No se encontraron archivos de datos
    echo 📁 Coloca uno de estos archivos en esta carpeta:
    echo    - DATA_GHAB.xlsx (formato Excel)
    echo    - paste.txt (formato texto)
    echo.
    set /p CONTINUAR="¿Deseas continuar sin datos? (s/n): "
    if /i not "!CONTINUAR!"=="s" (
        echo Operación cancelada
        pause
        exit /b 1
    )
)

echo.
echo ✅ Verificación completada
echo.

REM Crear estructura de carpetas
echo 📂 Creando estructura de proyecto...
if not exist "resultados" mkdir "resultados"
if not exist "resultados\dashboards" mkdir "resultados\dashboards"
if not exist "resultados\reportes" mkdir "resultados\reportes"
if not exist "resultados\graficos" mkdir "resultados\graficos"
echo ✓ Carpetas creadas

REM Preguntar al usuario si desea continuar
echo.
echo 🎯 LISTO PARA PROCEDER
echo.
echo El proceso incluye:
echo    1. 📦 Instalación automática de dependencias (5-15 min)
echo    2. 🚀 Ejecución del análisis completo (2-10 min)
echo    3. 📊 Generación de resultados interactivos
echo.
set /p CONTINUAR="¿Deseas continuar? (s/n): "
if /i not "%CONTINUAR%"=="s" (
    echo.
    echo ❌ Operación cancelada por el usuario
    pause
    exit /b 0
)

echo.
echo ================================================================================
echo 📦 FASE 1: INSTALACIÓN DE DEPENDENCIAS
echo ================================================================================
echo.

REM Instalar dependencias
echo 🔄 Ejecutando instalador automático...
python instalar_dependencias.py

if errorlevel 1 (
    echo.
    echo ❌ ERROR: La instalación de dependencias falló
    echo 💡 Intenta ejecutar manualmente:
    echo    pip install pandas numpy matplotlib seaborn plotly scikit-learn
    echo.
    set /p CONTINUAR_ERROR="¿Deseas intentar ejecutar el análisis de todos modos? (s/n): "
    if /i not "!CONTINUAR_ERROR!"=="s" (
        pause
        exit /b 1
    )
) else (
    echo.
    echo ✅ Dependencias instaladas correctamente
)

echo.
echo ================================================================================
echo 🚀 FASE 2: EJECUCIÓN DEL ANÁLISIS
echo ================================================================================
echo.

REM Ejecutar análisis
echo 📊 Iniciando análisis completo...
python ejecutar.py

if errorlevel 1 (
    echo.
    echo ❌ El análisis terminó con errores
    echo 💡 Revisa los mensajes anteriores para más información
    echo.
) else (
    echo.
    echo ✅ ANÁLISIS COMPLETADO EXITOSAMENTE!
    echo.
    echo 📁 Resultados guardados en carpeta 'resultados\'
    echo 🌐 Archivos HTML: Ábrelos en tu navegador
    echo 📊 Archivos CSV: Ábrelos en Excel
    echo 📄 Archivo TXT: Resumen ejecutivo
)

REM Preguntar si desea abrir la carpeta de resultados
echo.
set /p ABRIR_CARPETA="¿Deseas abrir la carpeta de resultados? (s/n): "
if /i "%ABRIR_CARPETA%"=="s" (
    if exist "resultados" (
        echo 📂 Abriendo carpeta de resultados...
        start "" "resultados"
    ) else (
        echo ⚠️ Carpeta de resultados no encontrada
    )
)

REM Preguntar si desea abrir el dashboard principal
echo.
if exist "resultados\dashboards\descriptivo_dashboard.html" (
    set /p ABRIR_DASHBOARD="¿Deseas abrir el dashboard principal en el navegador? (s/n): "
    if /i "!ABRIR_DASHBOARD!"=="s" (
        echo 🌐 Abriendo dashboard principal...
        start "" "resultados\dashboards\descriptivo_dashboard.html"
    )
)

echo.
echo ================================================================================
echo 🎉 PROCESO COMPLETADO
echo ================================================================================
echo.
echo 📊 Tu análisis de género y desarrollo económico está listo!
echo 🔍 Explora los resultados interactivos para descubrir insights únicos
echo 📈 Los gráficos HTML permiten zoom, filtros y exploración detallada
echo.
echo 💡 TIP: Guarda esta carpeta como respaldo de tu análisis
echo.

pause
exit /b 0

REM =====================================================================
REM FUNCIONES DE AYUDA
REM =====================================================================

:mostrar_ayuda
echo.
echo 📖 AYUDA - Análisis de Género y Desarrollo Económico
echo.
echo ARCHIVOS NECESARIOS:
echo   📄 analisis_principal.py     - Código principal del análisis
echo   📄 instalar_dependencias.py - Instalador automático
echo   📄 ejecutar.py              - Script de ejecución
echo   📄 requirements.txt         - Lista de dependencias
echo.
echo ARCHIVOS DE DATOS (uno de estos):
echo   📊 DATA_GHAB.xlsx           - Datos en formato Excel
echo   📊 paste.txt                - Datos en formato texto
echo.
echo PROBLEMAS COMUNES:
echo   ❓ Python no encontrado     - Instalar desde python.org
echo   ❓ Errores de permisos      - Ejecutar como administrador
echo   ❓ Archivos faltantes       - Verificar descargas completas
echo.
goto :eof

:limpiar_instalacion
echo.
echo 🧹 Limpiando instalación previa...
if exist "resultados" rmdir /s /q "resultados"
echo ✓ Limpieza completada
echo.
goto :eof
