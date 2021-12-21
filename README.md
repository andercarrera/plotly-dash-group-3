# plotly-dash-group-3
Autores:

## Introducción

### Conjunto de datos
El objetivo del dataset es ofrecer datos los cuales fueron recolectados en una sección de red de la Universidad del Cauca, Popayán, Colombia realizando capturas de paquetes a diferentes horas, durante la mañana y la tarde, durante seis días (26, 27, 28 de abril y 9, 11 y 15 de mayo) de 2017. Se recopilaron un total de 3.577.296 instancias y actualmente se almacenan en un archivo CSV (valores separados por comas).

Se ha juntado este dataset[1] con otro dataset[2] que proporciona información acerca de direcciones IP públicas con su respectiva geolocalización.

### Librerías
Todas las librerías utilizas están definidas en el fichero "requirements.txt".
```bash
pip install -r requirements.txt
```
## Elementos interactivos
1. Modificar la barrita que tenemos de KNN, pero en vez de eso que sea para el de Kmeans.

2. Correlación que ya está implementada 

3. Especificar ya sea con barra o con dropdown o escrito en un textbox (input type number) las dos variables de DBSCAN: eps y minPoints

4. Hacer un heatmap del dataset en un mapa mundi para ver de donde vienen las conexiones. Como elemento interactivo hemos pensado en un filtro (Dropdown) de algunos protocolos (ICMP...) o todos depende del curro que sea). Justificación: Como ejemplo en ICMP: Podemos ver desde dónde se hacen por lo tanto podemos ver tendencias como Ping Flood en este caso. (Los pings son ICMP)

## Algoritmos
1. DBSCAN
2. HDBSCAN
3. K-Means

## Filtros

### Filtro de algoritmos (dropdown simple)

### Filtro de instancias (dropdown simple)

### Filtro de correlaciones (dropdown múltiple)

## Preprocesamiento

## Referencias
1. [Dataset 1](https://datahub.io/core/geoip2-ipv4)
2. [Dataset 2](https://www.kaggle.com/jsrojas/ip-network-traffic-flows-labeled-with-87-apps)