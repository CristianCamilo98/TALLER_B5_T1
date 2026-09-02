import subprocess
print("=== Archivos identificados como LEGACY (individuales, pre-common-pipeline) ===\n")
print("Downstream/RMSE individual (superseded por common_pipeline/03_utility):")
for f in ["experimento_mezclas.py", "graficar_rmse_vs_sintetico.py", "analizar_coeficientes_ridge.py",
          "downstream_features.py", "verificar_supervisado.py", "construir_oracle.py"]:
    print(f"  generadores/marco/{f}")

print("\nFigures individuales (no generadas por common_pipeline/03_utility/plot_utility.py):")
result = subprocess.run(["powershell", "-Command", "Get-ChildItem generadores\\marco\\figures\\*.png | Select-Object -ExpandProperty Name"],
                         capture_output=True, text=True)
print(result.stdout)

print("Recomendacion: NO borrar todavia. El experimento de mezclas individual")
print("(RMSE 1.48 -> 0.26 con sintetico) sigue siendo la evidencia principal del")
print("README de Marco; common_pipeline/03_utility solo agrega comparabilidad")
print("entre generadores, no reemplaza el analisis individual. Revisar con el")
print("equipo antes de una limpieza futura.")
