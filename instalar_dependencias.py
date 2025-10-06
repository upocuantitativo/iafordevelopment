#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSTALADOR AUTOMÁTICO CORREGIDO - SIN PKG_RESOURCES
Análisis de Género y Desarrollo Económico

Versión simplificada que funciona en todas las versiones de Python
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Muestra el banner del instalador"""
    print("="*60)
    print("🔧 INSTALADOR AUTOMÁTICO - VERSIÓN CORREGIDA")
    print("📊 Análisis de Género y Desarrollo Económico")
    print("="*60)
    print()

def verificar_python():
    """Verifica que la versión de Python sea compatible"""
    version = sys.version_info
    print(f"🐍 Verificando Python...")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ ERROR: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        print(f"   Descarga Python desde: https://python.org")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def verificar_pip():
    """Verifica que pip esté disponible"""
    print(f"📦 Verificando pip...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ pip disponible: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ pip no funciona correctamente")
            return False
    except Exception as e:
        print(f"❌ Error verificando pip: {e}")
        return False

def actualizar_pip():
    """Actualiza pip a la última versión"""
    print(f"🔄 Actualizando pip...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✓ pip actualizado")
            return True
        else:
            print(f"⚠️ No se pudo actualizar pip (continuando de todos modos)")
            return True
    except Exception as e:
        print(f"⚠️ Error actualizando pip: {e}")
        return True  # Continuar de todos modos

def instalar_paquete(paquete, nombre_display=None):
    """Instala un paquete individual con manejo de errores robusto"""
    if nombre_display is None:
        nombre_display = paquete.split('>=')[0].split('==')[0]
    
    print(f"📦 Instalando {nombre_display}...", end=" ", flush=True)
    
    # Lista de métodos de instalación a probar
    metodos = [
        # Método 1: Instalación normal
        [sys.executable, "-m", "pip", "install", paquete],
        # Método 2: Con --user
        [sys.executable, "-m", "pip", "install", "--user", paquete],
        # Método 3: Forzar reinstalación
        [sys.executable, "-m", "pip", "install", "--force-reinstall", paquete],
        # Método 4: Sin dependencias (solo para casos extremos)
        [sys.executable, "-m", "pip", "install", "--no-deps", paquete],
    ]
    
    for i, metodo in enumerate(metodos):
        try:
            result = subprocess.run(
                metodo, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                sufijo = ""
                if "--user" in metodo:
                    sufijo = " (usuario)"
                elif "--force-reinstall" in metodo:
                    sufijo = " (reinstalado)"
                elif "--no-deps" in metodo:
                    sufijo = " (sin deps)"
                
                print(f"✓{sufijo}")
                return True
                
        except subprocess.TimeoutExpired:
            if i == len(metodos) - 1:  # Último intento
                print(f"❌ (timeout)")
                return False
            continue
        except Exception as e:
            if i == len(metodos) - 1:  # Último intento
                print(f"❌ (error)")
                return False
            continue
    
    print(f"❌")
    return False

def verificar_instalacion(paquete):
    """Verifica que un paquete se pueda importar"""
    try:
        __import__(paquete)
        return True
    except ImportError:
        return False

def instalar_dependencias_basicas():
    """Instala las dependencias más básicas primero"""
    print("\n🔧 INSTALANDO DEPENDENCIAS BÁSICAS")
    print("-" * 40)
    
    dependencias_basicas = [
        ("setuptools", "setuptools"),
        ("wheel", "wheel"),
        ("pip", "pip")
    ]
    
    for paquete, nombre in dependencias_basicas:
        instalar_paquete(paquete, nombre)
    
    print("✓ Dependencias básicas procesadas\n")

def instalar_dependencias_principales():
    """Instala las dependencias principales del proyecto"""
    print("📊 INSTALANDO DEPENDENCIAS PRINCIPALES")
    print("-" * 40)
    
    # Dependencias en orden de importancia
    dependencias = [
        # Núcleo esencial
        ("numpy>=1.21.0", "numpy"),
        ("pandas>=1.5.0", "pandas"),
        
        # Visualización básica
        ("matplotlib>=3.5.0", "matplotlib"),
        ("seaborn>=0.11.0", "seaborn"),
        
        # Excel
        ("openpyxl>=3.0.0", "openpyxl"),
        
        # Análisis científico
        ("scipy>=1.8.0", "scipy"),
        
        # Machine Learning
        ("scikit-learn>=1.1.0", "scikit-learn"),
        
        # Visualización interactiva
        ("plotly>=5.10.0", "plotly"),
        
        # Utilidades
        ("joblib>=1.1.0", "joblib"),
        ("tqdm>=4.64.0", "tqdm"),
    ]
    
    # Dependencias opcionales (no críticas)
    dependencias_opcionales = [
        ("kaleido>=0.2.1", "kaleido"),
        ("statsmodels>=0.13.0", "statsmodels"),
        ("xgboost>=1.6.0", "xgboost"),
    ]
    
    instalados = 0
    fallidos = []
    
    # Instalar dependencias principales
    for paquete, nombre in dependencias:
        if instalar_paquete(paquete, nombre):
            instalados += 1
        else:
            fallidos.append(nombre)
    
    # Instalar dependencias opcionales
    print(f"\n🔧 INSTALANDO DEPENDENCIAS OPCIONALES")
    print("-" * 40)
    
    for paquete, nombre in dependencias_opcionales:
        if instalar_paquete(paquete, nombre):
            instalados += 1
        else:
            fallidos.append(f"{nombre} (opcional)")
    
    return instalados, fallidos, len(dependencias) + len(dependencias_opcionales)

def verificar_imports():
    """Verifica que las librerías se puedan importar"""
    print(f"\n🔍 VERIFICANDO IMPORTACIONES")
    print("-" * 40)
    
    imports_criticos = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("sklearn", "scikit-learn"),
        ("plotly", "plotly"),
    ]
    
    imports_opcionales = [
        ("scipy", "scipy"),
        ("openpyxl", "openpyxl"),
        ("statsmodels", "statsmodels"),
        ("xgboost", "xgboost"),
    ]
    
    criticos_ok = 0
    opcionales_ok = 0
    
    print("Librerías críticas:")
    for modulo, nombre in imports_criticos:
        if verificar_instalacion(modulo):
            print(f"✓ {nombre}")
            criticos_ok += 1
        else:
            print(f"❌ {nombre}")
    
    print(f"\nLibrerías opcionales:")
    for modulo, nombre in imports_opcionales:
        if verificar_instalacion(modulo):
            print(f"✓ {nombre}")
            opcionales_ok += 1
        else:
            print(f"⚪ {nombre} (opcional)")
    
    return criticos_ok, len(imports_criticos), opcionales_ok, len(imports_opcionales)

def crear_estructura_proyecto():
    """Crea la estructura de directorios del proyecto"""
    print(f"\n📁 CREANDO ESTRUCTURA DEL PROYECTO")
    print("-" * 40)
    
    directorios = [
        "resultados",
        "resultados/dashboards",
        "resultados/reportes",
        "resultados/graficos",
    ]
    
    for directorio in directorios:
        try:
            Path(directorio).mkdir(parents=True, exist_ok=True)
            print(f"✓ {directorio}/")
        except Exception as e:
            print(f"❌ {directorio}/ - Error: {e}")
    
    # Crear archivo de instrucciones
    try:
        instrucciones = """# Instrucciones Rápidas

## Archivos de Datos Necesarios:
Coloca estos archivos en la carpeta principal:
- DATA_GHAB.xlsx (formato Excel)
- paste.txt (formato texto)

## Para Ejecutar el Análisis:
python ejecutar.py

## Resultados:
- Dashboards HTML interactivos en: resultados/dashboards/
- Reportes CSV/TXT en: resultados/reportes/

¡Los archivos HTML se abren en tu navegador web!
"""
        
        with open("INSTRUCCIONES.txt", "w", encoding="utf-8") as f:
            f.write(instrucciones)
        print(f"✓ INSTRUCCIONES.txt creado")
        
    except Exception as e:
        print(f"⚠️ No se pudo crear archivo de instrucciones: {e}")

def mostrar_resumen_final(criticos_ok, total_criticos, opcionales_ok, total_opcionales, fallidos):
    """Muestra el resumen final de la instalación"""
    print(f"\n" + "="*60)
    print(f"📊 RESUMEN FINAL DE INSTALACIÓN")
    print(f"="*60)
    
    print(f"✅ Librerías críticas: {criticos_ok}/{total_criticos}")
    print(f"🔧 Librerías opcionales: {opcionales_ok}/{total_opcionales}")
    
    if fallidos:
        print(f"\n⚠️ Paquetes que fallaron:")
        for paquete in fallidos:
            print(f"   - {paquete}")
    
    # Determinar estado del sistema
    if criticos_ok >= total_criticos * 0.8:  # 80% de críticos
        print(f"\n🎉 INSTALACIÓN EXITOSA!")
        print(f"✓ El análisis debería funcionar correctamente")
        
        print(f"\n📋 PRÓXIMOS PASOS:")
        print(f"1. Coloca tus archivos de datos en esta carpeta:")
        print(f"   - DATA_GHAB.xlsx")
        print(f"   - paste.txt")
        print(f"2. Ejecuta el análisis:")
        print(f"   python ejecutar.py")
        print(f"3. Los resultados aparecerán en resultados/")
        
        return True
    else:
        print(f"\n⚠️ INSTALACIÓN PARCIAL")
        print(f"Faltan demasiadas librerías críticas")
        
        print(f"\n💡 SOLUCIONES:")
        print(f"1. Intenta ejecutar de nuevo este instalador")
        print(f"2. Instala manualmente: pip install pandas numpy matplotlib")
        print(f"3. Considera usar Anaconda: https://anaconda.com")
        
        return False

def main():
    """Función principal del instalador corregido"""
    
    try:
        print_banner()
        
        # 1. Verificar Python
        if not verificar_python():
            input("\nPresiona Enter para salir...")
            return False
        
        # 2. Verificar pip
        if not verificar_pip():
            print("\n💡 Intentando reparar pip...")
            try:
                subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], 
                             capture_output=True)
                if not verificar_pip():
                    print("❌ No se pudo reparar pip")
                    input("\nPresiona Enter para salir...")
                    return False
            except:
                print("❌ No se pudo reparar pip")
                input("\nPresiona Enter para salir...")
                return False
        
        # 3. Actualizar pip
        actualizar_pip()
        
        # 4. Instalar dependencias básicas
        instalar_dependencias_basicas()
        
        # 5. Instalar dependencias principales
        instalados, fallidos, total = instalar_dependencias_principales()
        
        # 6. Verificar instalación
        criticos_ok, total_criticos, opcionales_ok, total_opcionales = verificar_imports()
        
        # 7. Crear estructura
        crear_estructura_proyecto()
        
        # 8. Mostrar resumen
        exito = mostrar_resumen_final(criticos_ok, total_criticos, opcionales_ok, total_opcionales, fallidos)
        
        input(f"\nPresiona Enter para continuar...")
        return exito
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Instalación cancelada por el usuario")
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print(f"💡 Intenta ejecutar: pip install pandas numpy matplotlib")
        input(f"\nPresiona Enter para salir...")
        return False

if __name__ == "__main__":
    exito = main()
    if not exito:
        sys.exit(1)