#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EJECUTOR PRINCIPAL
Análisis de Género y Desarrollo Económico

Este script ejecuta el análisis completo de manera simplificada,
verificando previamente que todo esté configurado correctamente.
"""

import sys
import os
import time
from datetime import datetime

def mostrar_banner():
    """Muestra el banner inicial"""
    print("\n" + "="*80)
    print("🚀 ANÁLISIS AVANZADO DE GÉNERO Y DESARROLLO ECONÓMICO")
    print("="*80)
    print("📊 Análisis para países de renta baja y media-baja")
    print("🔬 Técnicas: Correlaciones, Clustering, Machine Learning")
    print("📈 Resultados: Dashboards interactivos y reportes detallados")
    print("="*80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def verificar_python():
    """Verifica la versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    return True

def verificar_dependencias():
    """Verifica que las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...")
    
    dependencias_criticas = [
        ("pandas", "análisis de datos"),
        ("numpy", "cálculos numéricos"),
        ("matplotlib", "gráficos básicos"),
        ("plotly", "visualizaciones interactivas"),
        ("sklearn", "machine learning")
    ]
    
    faltantes = []
    
    for modulo, descripcion in dependencias_criticas:
        try:
            __import__(modulo)
            print(f"✓ {modulo} - {descripcion}")
        except ImportError:
            print(f"❌ {modulo} - {descripcion} (FALTANTE)")
            faltantes.append(modulo)
    
    if faltantes:
        print(f"\n⚠️ Faltan {len(faltantes)} dependencias críticas:")
        for modulo in faltantes:
            print(f"   - {modulo}")
        print("\n💡 Solución: Ejecuta primero 'python instalar_dependencias.py'")
        return False
    
    print("✓ Todas las dependencias están disponibles")
    return True

def verificar_archivos_datos():
    """Verifica que los archivos de datos estén disponibles"""
    print("\n📁 Verificando archivos de datos...")
    
    archivos_datos = [
        ("DATA_GHAB.xlsx", "archivo principal Excel"),
        ("paste.txt", "archivo alternativo texto")
    ]
    
    archivos_encontrados = []
    
    for archivo, descripcion in archivos_datos:
        if os.path.exists(archivo):
            size_mb = os.path.getsize(archivo) / (1024 * 1024)
            print(f"✓ {archivo} - {descripcion} ({size_mb:.1f} MB)")
            archivos_encontrados.append(archivo)
        else:
            print(f"❌ {archivo} - {descripcion} (NO ENCONTRADO)")
    
    if not archivos_encontrados:
        print("\n⚠️ No se encontraron archivos de datos")
        print("💡 Coloca uno de estos archivos en la carpeta:")
        print("   - DATA_GHAB.xlsx (formato Excel)")
        print("   - paste.txt (formato texto separado por tabs)")
        return False
    
    print(f"✓ Archivos de datos disponibles: {len(archivos_encontrados)}")
    return True

def verificar_estructura_proyecto():
    """Verifica/crea la estructura del proyecto"""
    print("\n📂 Verificando estructura del proyecto...")
    
    directorios_necesarios = [
        "resultados",
        "resultados/dashboards",
        "resultados/reportes"
    ]
    
    for directorio in directorios_necesarios:
        if not os.path.exists(directorio):
            try:
                os.makedirs(directorio, exist_ok=True)
                print(f"✓ Creado: {directorio}/")
            except Exception as e:
                print(f"❌ Error creando {directorio}: {e}")
                return False
        else:
            print(f"✓ Existe: {directorio}/")
    
    return True

def verificar_archivo_analisis():
    """Verifica que el archivo de análisis principal esté disponible"""
    print("\n📄 Verificando archivo de análisis...")
    
    if not os.path.exists("analisis_principal.py"):
        print("❌ Archivo 'analisis_principal.py' no encontrado")
        print("💡 Asegúrate de tener todos los archivos del proyecto")
        return False
    
    print("✓ Archivo de análisis principal disponible")
    return True

def estimar_tiempo_ejecucion():
    """Estima el tiempo de ejecución basado en los datos"""
    print("\n⏱️ Estimando tiempo de ejecución...")
    
    # Factores que afectan el tiempo
    factores = []
    
    # Tamaño de archivos
    for archivo in ["DATA_GHAB.xlsx", "paste.txt"]:
        if os.path.exists(archivo):
            size_mb = os.path.getsize(archivo) / (1024 * 1024)
            if size_mb > 5:  # Archivo grande
                factores.append("archivo_grande")
            break
    
    # Número estimado de variables (estimación por tamaño)
    tiempo_estimado = 2  # Base: 2 minutos
    
    if "archivo_grande" in factores:
        tiempo_estimado += 2
    
    print(f"🕒 Tiempo estimado: {tiempo_estimado}-{tiempo_estimado + 2} minutos")
    print("   (Depende de la velocidad de tu computadora)")
    
    return tiempo_estimado

def ejecutar_analisis():
    """Ejecuta el análisis principal"""
    print("\n" + "🚀 INICIANDO ANÁLISIS PRINCIPAL")
    print("="*50)
    
    try:
        # Importar y ejecutar el analizador
        from analisis_principal import AnalizadorGeneroDesarrollo
        
        print("📊 Creando instancia del analizador...")
        analizador = AnalizadorGeneroDesarrollo()
        
        print("🔄 Ejecutando análisis completo...")
        tiempo_inicio = time.time()
        
        exito = analizador.ejecutar_analisis_completo()
        
        tiempo_total = time.time() - tiempo_inicio
        
        if exito:
            print(f"\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
            print(f"⏱️ Tiempo total: {tiempo_total/60:.1f} minutos")
            return True
        else:
            print(f"\n❌ El análisis terminó con errores")
            print(f"⏱️ Tiempo transcurrido: {tiempo_total/60:.1f} minutos")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando el analizador: {e}")
        print("💡 Verifica que 'analisis_principal.py' esté en la carpeta")
        return False
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return False

def mostrar_resultados():
    """Muestra información sobre los resultados generados"""
    print("\n" + "="*60)
    print("📊 RESULTADOS GENERADOS")
    print("="*60)
    
    # Verificar archivos generados
    archivos_dashboards = [
        ("descriptivo_dashboard.html", "Estadísticas descriptivas"),
        ("top_correlaciones.html", "Variables más correlacionadas"),
        ("clusters_pca.html", "Agrupación de países"),
        ("comparacion_modelos_ml.html", "Modelos de Machine Learning")
    ]
    
    archivos_reportes = [
        ("estadisticas_descriptivas.csv", "Estadísticas de variables"),
        ("tabla_correlaciones.csv", "Todas las correlaciones"),
        ("perfiles_clusters.csv", "Perfiles de clusters"),
        ("resumen_ejecutivo.txt", "Resumen del análisis")
    ]
    
    print("📈 DASHBOARDS INTERACTIVOS (HTML):")
    dashboards_encontrados = 0
    for archivo, descripcion in archivos_dashboards:
        ruta_completa = os.path.join("resultados", "dashboards", archivo)
        if os.path.exists(ruta_completa):
            print(f"✓ {archivo} - {descripcion}")
            dashboards_encontrados += 1
        else:
            print(f"⚪ {archivo} - {descripcion} (no generado)")
    
    print(f"\n📋 REPORTES DE DATOS (CSV/TXT):")
    reportes_encontrados = 0
    for archivo, descripcion in archivos_reportes:
        ruta_completa = os.path.join("resultados", "reportes", archivo)
        if os.path.exists(ruta_completa):
            size_kb = os.path.getsize(ruta_completa) / 1024
            print(f"✓ {archivo} - {descripcion} ({size_kb:.1f} KB)")
            reportes_encontrados += 1
        else:
            print(f"⚪ {archivo} - {descripcion} (no generado)")
    
    print(f"\n📊 RESUMEN:")
    print(f"   Dashboards generados: {dashboards_encontrados}/{len(archivos_dashboards)}")
    print(f"   Reportes generados: {reportes_encontrados}/{len(archivos_reportes)}")
    
    if dashboards_encontrados > 0:
        print(f"\n🌐 CÓMO VER LOS RESULTADOS:")
        print(f"   1. Navega a la carpeta 'resultados/dashboards/'")
        print(f"   2. Haz doble clic en los archivos .html")
        print(f"   3. Se abrirán en tu navegador web")
        print(f"   4. Los gráficos son interactivos (puedes hacer zoom, filtrar, etc.)")
    
    if reportes_encontrados > 0:
        print(f"\n📄 REPORTES DE DATOS:")
        print(f"   - Los archivos .csv se pueden abrir en Excel")
        print(f"   - El archivo .txt contiene el resumen ejecutivo")

def main():
    """Función principal del ejecutor"""
    
    try:
        # Banner inicial
        mostrar_banner()
        
        # 1. Verificaciones previas
        print("🔍 VERIFICACIONES PREVIAS")
        print("-" * 30)
        
        if not verificar_python():
            input("\nPresiona Enter para salir...")
            return 1
        
        if not verificar_dependencias():
            input("\nPresiona Enter para salir...")
            return 1
        
        if not verificar_archivos_datos():
            input("\nPresiona Enter para salir...")
            return 1
        
        if not verificar_estructura_proyecto():
            input("\nPresiona Enter para salir...")
            return 1
        
        if not verificar_archivo_analisis():
            input("\nPresiona Enter para salir...")
            return 1
        
        # 2. Estimación de tiempo
        tiempo_estimado = estimar_tiempo_ejecucion()
        
        # 3. Confirmación del usuario
        print(f"\n✅ Todas las verificaciones pasaron correctamente")
        print(f"🎯 Listo para ejecutar el análisis completo")
        
        respuesta = input(f"\n¿Deseas continuar? (s/n): ").lower().strip()
        if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
            print("❌ Análisis cancelado por el usuario")
            return 0
        
        # 4. Ejecutar análisis
        exito = ejecutar_analisis()
        
        # 5. Mostrar resultados
        if exito:
            mostrar_resultados()
            
            print(f"\n🎊 ¡PROCESO COMPLETADO CON ÉXITO!")
            print(f"📁 Revisa la carpeta 'resultados/' para ver todos los archivos")
            print(f"🌟 ¡Disfruta explorando tus insights sobre género y desarrollo!")
            
            return 0
        else:
            print(f"\n❌ El análisis no se completó correctamente")
            print(f"💡 Revisa los mensajes de error anteriores")
            print(f"🔧 Puede que necesites reinstalar dependencias")
            
            return 1
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Análisis interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1
    finally:
        print(f"\n⏰ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    sys.exit(main())