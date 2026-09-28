# Datos

La aplicación no incluye los datasets originales del MINEDU. Los modelos joblib
permiten realizar inferencia sin volver a cargar los archivos de entrenamiento.

`ejemplo_entrada.csv` contiene únicamente tres registros ficticios con la
estructura admitida por la interfaz. Las categorías de `gestion` y `dsc_nivel`
proceden de los datos reales de 2023.

Para reentrenar, coloque los archivos 2023 y 2024 fuera del repositorio o indique
su carpeta mediante `--data-dir`.

