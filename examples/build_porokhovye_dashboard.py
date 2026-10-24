#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Сборка одного интерактивного дашборда по уже готовым данным ноутбука.

Что делает скрипт
-----------------
Читает файлы из ``examples/data/porokhovye/`` и собирает
``examples/dashboard.html`` — один файл, который открывается в браузере
двойным щелчком, без сервера и без Jupyter.

Скрипт **ничего не скачивает и не пересчитывает**:
* показатели кварталов уже лежат в ``processed/blocks_results.geojson``;
* сводные цифры — в ``processed/summary.json``;
* матрица времени — в ``processed/accessibility_matrix.csv``;
* школы и граница — в ``raw/``.

Он только:
1. упрощает геометрию, чтобы файл остался маленьким;
2. считает суммы, доли и гистограммы для графиков;
3. собирает карту, графики, таблицы и текст в один HTML.

Запуск (PowerShell, из корня проекта)
-------------------------------------
    .\\.venv\\Scripts\\python.exe examples\\build_porokhovye_dashboard.py

Результат: ``examples/dashboard.html`` (лимит — 20 МБ).
Для карты (тайлы) и графиков (Chart.js) браузеру нужен интернет.
Сам Python-скрипт интернет не использует.
"""

from __future__ import annotations

import bisect
import html as html_module
import json
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "porokhovye"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = BASE_DIR / "dashboard.html"

MAX_SIZE_MB = 20.0
SIMPLIFY_TOLERANCES_M = (3.0, 8.0, 20.0)
WORK_CRS = "EPSG:32636"

LAND_USE_RU = {
    "residential": "Жильё",
    "business": "Бизнес",
    "recreation": "Рекреация",
    "industrial": "Промышленность",
    "transport": "Транспорт",
    "special": "Специальная зона",
    "agriculture": "Сельское хозяйство",
    None: "Нет данных",
}

LAND_USE_COLORS = {
    "residential": "#f4a261",
    "business": "#e76f51",
    "recreation": "#2a9d8f",
    "industrial": "#8d6e63",
    "transport": "#6c757d",
    "special": "#9b59b6",
    "agriculture": "#a3b18a",
    None: "#dfe4ea",
}

MORPH_RU = {
    "low-rise low-density non-residential": "Малоэтажная, низкоплотная, нежилая",
    "high-mid-rise residential": "Средне-высотная жилая",
    "mid-rise low-density mixed-use": "Среднеэтажная низкоплотная, смешанная",
    "high-rise high-density residential": "Высотная высокоплотная жилая",
    "mid-rise low-density non-residential": "Среднеэтажная низкоплотная, нежилая",
}

STRELKA_RU = {
    "low-rise non-residential": "Малоэтажная нежилая",
    "mid-rise": "Среднеэтажная",
    "central model": "Центральная модель",
    "high-rise modern microdistrict": "Высотный современный микрорайон",
    "low-rise model": "Малоэтажная модель",
}

NO_DATA_COLOR = "#d5dbe1"
CATEGORY_PALETTE = [
    "#2a9d8f", "#e76f51", "#457b9d", "#e9c46a", "#8e7dbe",
    "#6c757d", "#a3b18a", "#bc6c25",
]

TOOLTIP_FIELDS = [
    "block_id",
    "land_use",
    "population",
    "schools",
    "demand",
    "provision",
    "density",
    "morph",
    "access",
]
TOOLTIP_ALIASES = [
    "Квартал",
    "Land use",
    "Население",
    "Школы",
    "Спрос",
    "Обеспеченность",
    "Плотность",
    "Морфотип",
    "Доступность",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def fail(message: str) -> None:
    print("\n[ОШИБКА] " + message)
    print("Дашборд не собран. Исправьте проблему и запустите скрипт снова.")
    raise SystemExit(1)


def esc(value: object) -> str:
    return html_module.escape(str(value), quote=True)


def is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fmt_num(value: object, digits: int = 0) -> str:
    """Число в русском виде: 136 053; 7,0; 29,1."""
    if is_blank(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number) or math.isinf(number):
        return "—"
    text = f"{number:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def fmt_pct(value: object, digits: int = 1) -> str:
    if is_blank(value):
        return "—"
    return fmt_num(float(value) * 100.0, digits) + " %"


def fmt_min(value: object) -> str:
    if is_blank(value):
        return "—"
    return fmt_num(value, 1) + " мин"


def round_coords(value: object, ndigits: int = 6) -> object:
    """Округляет координаты GeoJSON до ~0,1 м, чтобы файл не разрастался."""
    if isinstance(value, (list, tuple)):
        return [round_coords(item, ndigits) for item in value]
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def quantile_edges(values: list[float], classes: int = 5) -> list[float]:
    clean = [float(v) for v in values if not is_blank(v) and math.isfinite(float(v))]
    if not clean:
        return []
    quantiles = np.quantile(np.asarray(clean), np.linspace(0.0, 1.0, classes + 1))
    edges: list[float] = []
    for q in quantiles:
        q = round(float(q), 6)
        if not edges or q > edges[-1]:
            edges.append(q)
    return edges


def color_by_edges(value: object, edges: list[float], palette: list[str]) -> str:
    if is_blank(value) or not edges:
        return NO_DATA_COLOR
    index = bisect.bisect_right(edges, float(value)) - 1
    index = max(0, min(index, len(palette) - 1))
    return palette[index]


def html_table(
    headers: list[str],
    rows: list[list[str]],
    table_id: str | None = None,
    numeric_from: int | None = None,
) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body_parts: list[str] = []
    for row in rows:
        cells: list[str] = []
        for index, cell in enumerate(row):
            css = ' class="num"' if numeric_from is not None and index >= numeric_from else ""
            cells.append(f"<td{css}>{cell}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    id_attr = f' id="{table_id}"' if table_id else ""
    return (
        f'<table{id_attr} class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_parts)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Чтение исходных файлов
# ---------------------------------------------------------------------------


def read_inputs() -> dict[str, object]:
    required = {
        "blocks_results.geojson": PROCESSED_DIR / "blocks_results.geojson",
        "summary.json": PROCESSED_DIR / "summary.json",
        "accessibility_matrix.csv": PROCESSED_DIR / "accessibility_matrix.csv",
        "school.geojson": RAW_DIR / "school.geojson",
        "boundary.geojson": RAW_DIR / "boundary.geojson",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        fail(
            "Не найдены файлы: "
            + ", ".join(missing)
            + ".\nСначала полностью выполните ноутбук examples\\porokhovye_research.ipynb."
        )

    blocks = gpd.read_file(required["blocks_results.geojson"])
    if blocks.empty:
        fail("Файл blocks_results.geojson пустой.")
    if "block_id" not in blocks.columns or "geometry" not in blocks.columns:
        fail("В blocks_results.geojson нет колонок block_id / geometry.")

    summary = json.loads(required["summary.json"].read_text(encoding="utf-8"))

    matrix = pd.read_csv(required["accessibility_matrix.csv"], index_col=0)
    if matrix.empty:
        fail("Матрица accessibility_matrix.csv пустая.")

    schools = gpd.read_file(required["school.geojson"])
    boundary = gpd.read_file(required["boundary.geojson"])

    for frame in (schools, boundary):
        if frame.crs is None:
            frame.set_crs("EPSG:4326", inplace=True)

    blocks_schools = None
    path = PROCESSED_DIR / "blocks_schools.geojson"
    if path.exists():
        blocks_schools = gpd.read_file(path)

    graph_edges = None
    path = PROCESSED_DIR / "graph_edges.geojson"
    if path.exists():
        graph_edges = gpd.read_file(path)

    links = None
    path = PROCESSED_DIR / "provision_links.csv"
    if path.exists():
        try:
            links = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            links = pd.DataFrame()

    return {
        "blocks": blocks,
        "summary": summary,
        "matrix": matrix,
        "schools": schools,
        "boundary": boundary,
        "blocks_schools": blocks_schools,
        "graph_edges": graph_edges,
        "links": links,
        "summary_mtime": required["summary.json"].stat().st_mtime,
    }


# ---------------------------------------------------------------------------
# Подготовка блоков
# ---------------------------------------------------------------------------


def prepare_blocks(blocks: gpd.GeoDataFrame, tolerance_m: float) -> gpd.GeoDataFrame:
    gdf = blocks.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(WORK_CRS)
    gdf = gdf.to_crs(WORK_CRS)
    gdf["geometry"] = gdf.geometry.simplify(tolerance_m, preserve_topology=True)
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def translate_morph(value: object) -> str | None:
    if is_blank(value):
        return None
    text = str(value)
    return MORPH_RU.get(text, text)


def translate_strelka(value: object) -> str | None:
    if is_blank(value):
        return None
    text = str(value)
    return STRELKA_RU.get(text, text)


def base_properties(row: pd.Series) -> dict[str, object]:
    land_use_key = None if is_blank(row.get("land_use")) else str(row.get("land_use"))
    population = 0.0 if is_blank(row.get("population")) else float(row["population"])
    capacity = 0.0 if is_blank(row.get("school_capacity")) else float(row["school_capacity"])
    demand = 0.0 if is_blank(row.get("demand")) else float(row["demand"])
    provision = row.get("provision_strong")
    fsi = row.get("fsi")
    gsi = row.get("gsi")
    access = row.get("area_accessibility")

    schools_text = f"{fmt_num(capacity)} мест" if capacity > 0 else "нет"
    demand_text = f"{fmt_num(demand)} мест" if demand > 0 else "нет спроса"
    if demand <= 0:
        provision_text = "нет жителей"
    elif is_blank(provision):
        provision_text = "—"
    else:
        provision_text = fmt_pct(provision)

    if is_blank(fsi) or is_blank(gsi):
        density_text = "—"
    else:
        density_text = f"FSI {fmt_num(fsi, 2)} · GSI {fmt_num(gsi, 2)}"

    morph = translate_morph(row.get("spacematrix_morphotype")) or "—"

    return {
        "block_id": int(row["block_id"]),
        "land_use": LAND_USE_RU.get(land_use_key, "Нет данных"),
        "population": f"{fmt_num(population)} чел.",
        "schools": schools_text,
        "demand": demand_text,
        "provision": provision_text,
        "density": density_text,
        "morph": morph,
        "access": fmt_min(access),
    }


def build_features(gdf: gpd.GeoDataFrame) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []
    for _, row in gdf.iterrows():
        geometry = mapping(row.geometry)
        features.append(
            {
                "type": "Feature",
                "properties": base_properties(row),
                "geometry": {
                    "type": geometry["type"],
                    "coordinates": round_coords(geometry["coordinates"]),
                },
            }
        )
    return features


# ---------------------------------------------------------------------------
# Слои карты
# ---------------------------------------------------------------------------


def numeric_metric(
    key: str,
    label: str,
    column: str,
    palette: list[str],
    caption: str,
    digits: int = 2,
    unit: str = "",
    zero_as_no_data: bool = False,
) -> dict[str, object]:
    return {
        "key": key,
        "label": label,
        "kind": "numeric",
        "column": column,
        "palette": palette,
        "caption": caption,
        "digits": digits,
        "unit": unit,
        "show": False,
        "zero_as_no_data": zero_as_no_data,
    }


def categorical_metric(
    key: str,
    label: str,
    column: str,
    translate,
    color_map: dict[object, str],
    caption: str,
    show: bool = False,
) -> dict[str, object]:
    return {
        "key": key,
        "label": label,
        "kind": "categorical",
        "column": column,
        "translate": translate,
        "color_map": color_map,
        "caption": caption,
        "show": show,
    }


def metric_configs(gdf: gpd.GeoDataFrame) -> list[dict[str, object]]:
    def land_use_value(value: object) -> str:
        key = None if is_blank(value) else str(value)
        return LAND_USE_RU.get(key, "Нет данных")

    def morph_value(value: object) -> str:
        return translate_morph(value) or "Нет морфотипа"

    def strelka_value(value: object) -> str:
        return translate_strelka(value) or "Нет морфотипа"

    morph_values = sorted({morph_value(v) for v in gdf["spacematrix_morphotype"]})
    morph_values = [v for v in morph_values if v != "Нет морфотипа"]
    morph_colors = {
        name: CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)]
        for index, name in enumerate(morph_values)
    }
    morph_colors["Нет морфотипа"] = NO_DATA_COLOR

    strelka_values = sorted({strelka_value(v) for v in gdf["strelka_morphotype"]})
    strelka_values = [v for v in strelka_values if v != "Нет морфотипа"]
    strelka_colors = {
        name: CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)]
        for index, name in enumerate(strelka_values)
    }
    strelka_colors["Нет морфотипа"] = NO_DATA_COLOR

    return [
        {
            "key": "land_use",
            "label": "Land use",
            "kind": "categorical",
            "column": "land_use",
            "translate": land_use_value,
            "color_map": {LAND_USE_RU[key]: color for key, color in LAND_USE_COLORS.items()},
            "caption": "Назначение квартала по OSM landuse.",
            "show": True,
        },
        numeric_metric(
            "population",
            "Население, чел.",
            "population",
            ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#8c2d04"],
            "Модельная оценка: жилая площадь / 30 м² на человека.",
            digits=0,
            zero_as_no_data=True,
        ),
        {
            "key": "provision",
            "label": "Обеспеченность школами",
            "kind": "provision",
            "column": "provision_strong",
            "palette": ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"],
            "edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001],
            "caption": "Доля спроса, покрытая своими местами. Серый — нет жителей.",
            "digits": 1,
        },
        numeric_metric(
            "fsi",
            "FSI — плотность застройки",
            "fsi",
            ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
            "Площадь этажей / площадь квартала. 0 — застройки нет.",
            zero_as_no_data=True,
        ),
        numeric_metric(
            "gsi",
            "GSI — плотность пятна",
            "gsi",
            ["#f7fcfd", "#d9f0d3", "#a6dba0", "#4d9221", "#00441b"],
            "Площадь застройки / площадь квартала. Больше — меньше дворов.",
            zero_as_no_data=True,
        ),
        numeric_metric(
            "area_accessibility",
            "Площадная доступность, мин",
            "area_accessibility",
            ["#fff7fb", "#d0d1e6", "#74a9cf", "#0570b0", "#023858"],
            "Среднее время до всех кварталов округа. Темнее — дальше.",
            unit=" мин",
        ),
        categorical_metric(
            "spacematrix",
            "Морфотип Spacematrix",
            "spacematrix_morphotype",
            morph_value,
            morph_colors,
            "Тип застройки по высотности, плотности и доле жилья.",
        ),
        categorical_metric(
            "strelka",
            "Морфотип Strelka",
            "strelka_morphotype",
            strelka_value,
            strelka_colors,
            "Альтернативная типология застройки.",
        ),
    ]


def build_map(
    gdf: gpd.GeoDataFrame,
    features: list[dict[str, object]],
    boundary: gpd.GeoDataFrame,
    schools: gpd.GeoDataFrame,
    graph_edges: gpd.GeoDataFrame | None,
) -> tuple[str, str]:
    import folium

    boundary_wgs = boundary.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = boundary_wgs.total_bounds
    center = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]

    fmap = folium.Map(
        location=center,
        zoom_start=13,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    # Подложка Esri Light Gray: тайлы CARTO требуют API-ключ («API KEY REQUIRED»),
    # а сервер OpenStreetMap блокирует страницы, открытые как файл (403 Access blocked).
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
        name="Подложка Esri (светлая)",
        max_zoom=16,
    ).add_to(fmap)

    folium.GeoJson(
        boundary_wgs,
        name="Граница округа",
        style_function=lambda _: {
            "color": "#12324f",
            "weight": 3,
            "fill": False,
        },
    ).add_to(fmap)

    for metric in metric_configs(gdf):
        column = metric["column"]
        if metric["kind"] == "numeric":
            values = [
                None if is_blank(v) else float(v)
                for v in gdf[column].tolist()
            ]
            zero_as_no_data = bool(metric.get("zero_as_no_data"))
            classified = [
                value
                for value in values
                if value is not None and not (zero_as_no_data and value <= 0)
            ]
            edges = quantile_edges(classified, 5)
            palette = metric["palette"]
            metric["edges"] = edges

            def style_function(
                feature,
                _palette=palette,
                _edges=edges,
                _zero=zero_as_no_data,
            ):
                value = feature["properties"].get("value")
                if _zero and value is not None and float(value) <= 0:
                    fill = NO_DATA_COLOR
                else:
                    fill = color_by_edges(value, _edges, _palette)
                return {
                    "fillColor": fill,
                    "color": "#ffffff",
                    "weight": 0.7,
                    "fillOpacity": 0.78,
                }

        elif metric["kind"] == "provision":
            edges = metric["edges"]
            palette = metric["palette"]

            def style_function(feature, _palette=palette, _edges=edges):
                return {
                    "fillColor": color_by_edges(feature["properties"].get("value"), _edges, _palette),
                    "color": "#ffffff",
                    "weight": 0.7,
                    "fillOpacity": 0.82,
                }

        else:
            color_map = metric["color_map"]

            def style_function(feature, _colors=color_map):
                value = feature["properties"].get("value")
                return {
                    "fillColor": _colors.get(value, NO_DATA_COLOR),
                    "color": "#ffffff",
                    "weight": 0.7,
                    "fillOpacity": 0.8,
                }

        layer_features: list[dict[str, object]] = []
        for feature, (_, row) in zip(features, gdf.iterrows()):
            properties = dict(feature["properties"])

            if metric["kind"] == "numeric":
                raw = row[column]
                properties["value"] = None if is_blank(raw) else float(raw)
            elif metric["kind"] == "provision":
                raw = row[column]
                properties["value"] = None if is_blank(raw) else float(raw)
            else:
                properties["value"] = metric["translate"](row[column])

            layer_features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": feature["geometry"],
                }
            )

        folium.GeoJson(
            {"type": "FeatureCollection", "features": layer_features},
            name=metric["label"],
            style_function=style_function,
            highlight_function=lambda _: {"weight": 2.0, "color": "#12324f"},
            tooltip=folium.GeoJsonTooltip(
                fields=TOOLTIP_FIELDS,
                aliases=TOOLTIP_ALIASES,
                labels=True,
                sticky=False,
                style=(
                    "background:#ffffff;border:1px solid #cfd8e3;border-radius:10px;"
                    "box-shadow:0 8px 24px rgba(18,50,79,.18);padding:10px 12px;"
                    "font-size:12px;color:#16202b;"
                ),
            ),
            show=metric.get("show", False),
        ).add_to(fmap)

    if graph_edges is not None and not graph_edges.empty:
        folium.GeoJson(
            graph_edges.to_crs("EPSG:4326"),
            name="Уличная сеть графа (авто)",
            style_function=lambda _: {"color": "#7b8794", "weight": 1, "opacity": 0.55},
            show=False,
        ).add_to(fmap)

    schools_wgs = schools.to_crs("EPSG:4326")
    school_group = folium.FeatureGroup(name=f"Школы ({len(schools_wgs)})", show=True)
    for _, school in schools_wgs.iterrows():
        point = school.geometry.representative_point()
        name = school.get("name")
        name = "Школа" if is_blank(name) else str(name)
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=5,
            color="#7f1d1d",
            weight=1.5,
            fill=True,
            fill_color="#e63946",
            fill_opacity=0.9,
            tooltip=esc(name),
        ).add_to(school_group)
    school_group.add_to(fmap)

    # Свёрнутый список слоёв (раскрывается при наведении): развёрнутый был выше карты
    # и прятался под липким меню разделов.
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])

    rendered = fmap.get_root().render()
    head = rendered.split("<head>", 1)[1].split("</head>", 1)[0]
    # folium кладёт JavaScript карты (L.map, слои GeoJSON) ПОСЛЕ </body>,
    # поэтому берём всё после <body> до </html>, иначе карта остаётся пустой.
    body = rendered.split("<body>", 1)[1].replace("</body>", "", 1).rsplit("</html>", 1)[0]
    return head, body


# ---------------------------------------------------------------------------
# Данные для графиков
# ---------------------------------------------------------------------------


def chart_configs(
    gdf: gpd.GeoDataFrame,
    matrix: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, object]:
    charts: dict[str, object] = {}

    # 1. Структура площади по land use.
    lu = gdf.copy()
    lu["_key"] = lu["land_use"].where(lu["land_use"].notna(), None)
    grouped = lu.groupby("_key", dropna=False)["site_area"].sum().sort_values(ascending=False)
    total_area = float(grouped.sum())
    labels = [LAND_USE_RU.get(None if is_blank(k) else str(k), "Нет данных") for k in grouped.index]
    values = [round(float(v) / total_area * 100.0, 1) for v in grouped.values]
    colors = [LAND_USE_COLORS.get(None if is_blank(k) else str(k), NO_DATA_COLOR) for k in grouped.index]
    charts["chartLandUse"] = {
        "type": "doughnut",
        "data": {
            "labels": labels,
            "datasets": [{"data": values, "backgroundColor": colors, "borderColor": "#ffffff", "borderWidth": 2}],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "cutout": "55%",
            "plugins": {
                "legend": {"position": "right", "labels": {"boxWidth": 12, "font": {"size": 11}}},
            },
        },
    }

    # 2. Население по кварталам (только жилые).
    pop = gdf[gdf["population"].fillna(0) > 0].sort_values("population", ascending=False)
    pop_labels = [f"Кв. {int(v)}" for v in pop["block_id"]]
    charts["chartPopulation"] = {
        "type": "bar",
        "data": {
            "labels": pop_labels,
            "datasets": [
                {
                    "label": "Население, чел.",
                    "data": [int(v) for v in pop["population"]],
                    "backgroundColor": "#e76f51",
                    "borderRadius": 4,
                }
            ],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"beginAtZero": True, "grid": {"color": "#eef1f5"}},
                "y": {"grid": {"display": False}, "ticks": {"font": {"size": 10}}},
            },
        },
    }

    # 3. Обеспеченность по кварталам.
    prov = gdf[gdf["demand"].fillna(0) > 0].sort_values("provision_strong", ascending=True)
    prov_labels = [f"Кв. {int(v)}" for v in prov["block_id"]]
    prov_values = [round(float(v) * 100.0, 1) if not is_blank(v) else 0.0 for v in prov["provision_strong"]]
    prov_colors = ["#b42318" if v < 30 else "#d97706" if v < 60 else "#2a9d8f" for v in prov_values]
    charts["chartProvision"] = {
        "type": "bar",
        "data": {
            "labels": prov_labels,
            "datasets": [
                {
                    "label": "Обеспеченность, %",
                    "data": prov_values,
                    "backgroundColor": prov_colors,
                    "borderRadius": 4,
                }
            ],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {
                    "beginAtZero": True,
                    "max": 100,
                    "grid": {"color": "#eef1f5"},
                    "title": {"display": True, "text": "% спроса"},
                },
                "y": {"grid": {"display": False}, "ticks": {"font": {"size": 10}}},
            },
        },
    }

    # 4. Вместимость и спрос по кварталам.
    both = gdf[gdf["demand"].fillna(0) > 0].sort_values("demand", ascending=False)
    both_labels = [f"Кв. {int(v)}" for v in both["block_id"]]
    charts["chartCapacity"] = {
        "type": "bar",
        "data": {
            "labels": both_labels,
            "datasets": [
                {
                    "label": "Спрос, мест",
                    "data": [int(v) for v in both["demand"]],
                    "backgroundColor": "#457b9d",
                    "borderRadius": 3,
                },
                {
                    "label": "Вместимость (модель), мест",
                    "data": [int(v) for v in both["school_capacity"]],
                    "backgroundColor": "#e9c46a",
                    "borderRadius": 3,
                },
            ],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"position": "top", "labels": {"boxWidth": 12, "font": {"size": 10}}}},
            "scales": {
                "x": {"grid": {"display": False}, "ticks": {"font": {"size": 9}}},
                "y": {"beginAtZero": True, "grid": {"color": "#eef1f5"}},
            },
        },
    }

    # 5. Гистограмма FSI.
    fsi_edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    fsi_labels = ["0 (нет застройки)", "0–0,25", "0,25–0,5", "0,5–0,75", "0,75–1", "1–1,5", "1,5–2", "2–3"]
    fsi_values = gdf["fsi"].fillna(0.0).tolist()
    fsi_counts = [0] * len(fsi_labels)
    for value in fsi_values:
        number = float(value)
        if number <= 0:
            fsi_counts[0] += 1
            continue
        index = bisect.bisect_right(fsi_edges, number) - 1
        index = max(1, min(index, len(fsi_counts) - 1))
        fsi_counts[index] += 1
    charts["chartFsi"] = {
        "type": "bar",
        "data": {
            "labels": fsi_labels,
            "datasets": [{"label": "Кварталов", "data": fsi_counts, "backgroundColor": "#2171b5", "borderRadius": 3}],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"grid": {"display": False}, "title": {"display": True, "text": "FSI"}},
                "y": {"beginAtZero": True, "grid": {"color": "#eef1f5"}, "title": {"display": True, "text": "кварталов"}},
            },
        },
    }

    # 6. Гистограмма времени в пути по матрице.
    values = matrix.to_numpy(dtype=float)
    n = values.shape[0]
    if n:
        mask = ~np.eye(n, dtype=bool)
        flat = values[mask]
        flat = flat[np.isfinite(flat)]
    else:
        flat = np.array([], dtype=float)
    if flat.size:
        time_upper = max(2.0, math.ceil(float(flat.max()) / 2.0) * 2.0)
    else:
        time_upper = 14.0
    bins = [float(x) for x in np.arange(0.0, time_upper + 2.0, 2.0)]
    time_labels = [f"{bins[i]:g}–{bins[i + 1]:g}" for i in range(len(bins) - 1)]
    time_counts = [0] * len(time_labels)
    for value in flat:
        index = bisect.bisect_right(bins, float(value)) - 1
        index = max(0, min(index, len(time_counts) - 1))
        time_counts[index] += 1
    charts["chartTravel"] = {
        "type": "bar",
        "data": {
            "labels": time_labels,
            "datasets": [{"label": "Пар кварталов", "data": time_counts, "backgroundColor": "#2a9d8f", "borderRadius": 3}],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"grid": {"display": False}, "title": {"display": True, "text": "минут"}},
                "y": {"beginAtZero": True, "grid": {"color": "#eef1f5"}},
            },
        },
    }

    # 7. Площадная доступность.
    acc_values = [float(v) for v in gdf["area_accessibility"].dropna().tolist()]
    if acc_values:
        acc_low = math.floor(min(acc_values))
        acc_high = math.ceil(max(acc_values))
        if acc_high - acc_low < 2:
            acc_high = acc_low + 2
        acc_step = 0.5 if (acc_high - acc_low) <= 3 else 1.0
        acc_bins = [
            float(x) for x in np.arange(acc_low, acc_high + acc_step * 0.5, acc_step)
        ]
    else:
        acc_bins = [7.0, 8.0, 9.0, 10.0]
    acc_labels = [
        f"{acc_bins[i]:g}–{acc_bins[i + 1]:g}".replace(".", ",")
        for i in range(len(acc_bins) - 1)
    ]
    acc_counts = [0] * len(acc_labels)
    for value in acc_values:
        index = bisect.bisect_right(acc_bins, float(value)) - 1
        index = max(0, min(index, len(acc_counts) - 1))
        acc_counts[index] += 1
    charts["chartAccess"] = {
        "type": "bar",
        "data": {
            "labels": acc_labels,
            "datasets": [{"label": "Кварталов", "data": acc_counts, "backgroundColor": "#457b9d", "borderRadius": 3}],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"grid": {"display": False}, "title": {"display": True, "text": "минут"}},
                "y": {"beginAtZero": True, "grid": {"color": "#eef1f5"}},
            },
        },
    }

    # 8. Морфотипы Spacematrix.
    morph_counts: dict[str, int] = {}
    for value in gdf["spacematrix_morphotype"]:
        if is_blank(value):
            continue
        label = translate_morph(value) or "—"
        morph_counts[label] = morph_counts.get(label, 0) + 1
    morph_sorted = sorted(morph_counts.items(), key=lambda item: item[1], reverse=True)
    charts["chartMorph"] = {
        "type": "bar",
        "data": {
            "labels": [item[0] for item in morph_sorted],
            "datasets": [
                {
                    "label": "Кварталов",
                    "data": [item[1] for item in morph_sorted],
                    "backgroundColor": "#8e7dbe",
                    "borderRadius": 4,
                }
            ],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"beginAtZero": True, "grid": {"color": "#eef1f5"}, "ticks": {"precision": 0}},
                "y": {"grid": {"display": False}, "ticks": {"font": {"size": 10}}},
            },
        },
    }

    # 9. Морфотипы Strelka.
    strelka_counts: dict[str, int] = {}
    for value in gdf["strelka_morphotype"]:
        if is_blank(value):
            continue
        label = translate_strelka(value) or "—"
        strelka_counts[label] = strelka_counts.get(label, 0) + 1
    strelka_sorted = sorted(strelka_counts.items(), key=lambda item: item[1], reverse=True)
    charts["chartStrelka"] = {
        "type": "bar",
        "data": {
            "labels": [item[0] for item in strelka_sorted],
            "datasets": [
                {
                    "label": "Кварталов",
                    "data": [item[1] for item in strelka_sorted],
                    "backgroundColor": "#e9c46a",
                    "borderRadius": 4,
                }
            ],
        },
        "options": {
            "indexAxis": "y",
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"beginAtZero": True, "grid": {"color": "#eef1f5"}, "ticks": {"precision": 0}},
                "y": {"grid": {"display": False}, "ticks": {"font": {"size": 10}}},
            },
        },
    }

    return charts


# ---------------------------------------------------------------------------
# Текстовые блоки
# ---------------------------------------------------------------------------


def build_kpi(summary: dict[str, object], gdf: gpd.GeoDataFrame) -> str:
    territory = summary.get("territory", {})
    population = summary.get("population", {})
    schools = summary.get("schools", {})
    travel = summary.get("travel_time", {})
    blocks = summary.get("blocks", {})

    area = territory.get("area_km2", gdf["site_area"].sum() / 1e6)
    cards = [
        ("Площадь округа", fmt_num(area, 2) + " км²", "OSM R1185384"),
        ("Кварталов", fmt_num(blocks.get("count", len(gdf))), "после объединения «пылинок»"),
        ("Население (модель)", fmt_num(population.get("total", gdf["population"].sum())), "не официальная статистика"),
        ("Плотность (модель)", fmt_num(population.get("density_per_km2"), 0), "чел./км²"),
        ("Школ", fmt_num(schools.get("count", 19)), "по данным OSM"),
        ("Вместимость (модель)", fmt_num(schools.get("capacity_total")), "250 мест на школу"),
        ("Спрос", fmt_num(schools.get("demand_total")), "120 мест на 1000 жителей"),
        ("Обеспеченность", fmt_pct(schools.get("provision_strong")), "артефакт, см. ниже"),
        ("Медианное время", fmt_num(travel.get("median_min"), 1) + " мин", "авто, без пробок"),
        ("90-й процентиль", fmt_num(travel.get("p90_min"), 1) + " мин", "недостижимых пар: " + fmt_pct(travel.get("share_unreachable"))),
    ]
    cards_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
        for label, value, sub in cards
    )
    return f'<div class="kpi-grid">{cards_html}</div>'


def build_legend(gdf: gpd.GeoDataFrame) -> str:
    items = [
        '<div class="legend-item"><span class="swatch" style="background:#d73027"></span>'
        "<span><b>Обеспеченность:</b> красный — 0 %, жёлтый — около 40–60 %, зелёный — 80–100 %. "
        "Серый — в квартале нет жителей, поэтому нет спроса.</span></div>",
        '<div class="legend-item"><span class="swatch" style="background:#e76f51"></span>'
        "<span><b>Население:</b> от светло-жёлтого (мало) к тёмно-коричневому (много).</span></div>",
        '<div class="legend-item"><span class="swatch" style="background:#2171b5"></span>'
        "<span><b>FSI / GSI / доступность:</b> чем темнее, тем больше значение. "
        "Для доступности темнее — дальше.</span></div>",
        '<div class="legend-item"><span class="swatch" style="background:#2a9d8f"></span>'
        "<span><b>Морфотипы и land use:</b> цвета категорий; точное значение — в подсказке при наведении.</span></div>",
    ]
    return '<div class="legend-grid">' + "".join(items) + "</div>"


def build_summary_table(summary: dict[str, object], gdf: gpd.GeoDataFrame) -> str:
    territory = summary.get("territory", {})
    graph = summary.get("graph", {})
    blocks = summary.get("blocks", {})
    population = summary.get("population", {})
    schools = summary.get("schools", {})
    travel = summary.get("travel_time", {})

    rows = [
        ["Территория", esc(territory.get("name", "Пороховые")), "OSM R1185384"],
        ["Площадь округа", fmt_num(territory.get("area_km2"), 3) + " км²", "по границе OSM"],
        ["Кварталов", fmt_num(blocks.get("count", len(gdf))), "итоговая нарезка"],
        ["Медианная площадь квартала", fmt_num(blocks.get("median_area_m2"), 0) + " м²", "модельная нарезка"],
        ["Доля застроенных кварталов", fmt_pct(blocks.get("built_share")), "FSI > 0"],
        ["Население (модель)", fmt_num(population.get("total")) + " чел.", "жилая площадь / 30 м²"],
        ["Плотность (модель)", fmt_num(population.get("density_per_km2")) + " чел./км²", "на площадь округа"],
        ["Школ", fmt_num(schools.get("count")), "по данным OSM"],
        ["Вместимость (модель)", fmt_num(schools.get("capacity_total")) + " мест", "250 мест на школу"],
        ["Норматив спроса", fmt_num(schools.get("demand_per_1000")) + " мест / 1000 жителей", "конфигурация BlocksNet"],
        ["Расчётный спрос", fmt_num(schools.get("demand_total")) + " мест", "население × норматив"],
        ["Обеспеченность strong", fmt_pct(schools.get("provision_strong")), "доля покрытого спроса"],
        ["Обеспеченность weak", fmt_pct(schools.get("provision_weak")), "совпадает со strong (глубина 1)"],
        ["Медианное время", fmt_num(travel.get("median_min"), 2) + " мин", "автомобиль, без пробок"],
        ["90-й процентиль времени", fmt_num(travel.get("p90_min"), 2) + " мин", "90 % пар быстрее"],
        ["Максимальное время", fmt_num(travel.get("max_min"), 2) + " мин", "в пределах округа"],
        ["Недостижимые пары", fmt_pct(travel.get("share_unreachable")), "0 % — граф связный"],
        ["Узлов / рёбер графа", f"{fmt_num(graph.get('nodes'))} / {fmt_num(graph.get('edges'))}", "крупнейшая связная компонента"],
    ]
    return html_table(["Показатель", "Значение", "Комментарий"], rows)


def build_territory_notes() -> str:
    return """
    <ul class="notes">
      <li><b>Парк Малиновка</b> и <b>парк у Охты</b> отнесены к <b>рекреации</b>:
          в OSM они размечены как парки, и это совпадает с визуальной проверкой.</li>
      <li><b>Хасанский рынок</b> и <b>«Лента»</b> у железной дороги — <b>бизнес</b>:
          это торговые объекты, а не жильё и не склады.</li>
      <li><b>Автобусный парк</b> справа на карте выглядит <b>промышленным</b>.
          В OSM участок размечен <code>landuse=industrial</code>, а не <code>depot</code>,
          поэтому правило «депо → транспорт» не сработало. Мы оставили класс как есть:
          на население и школы это не влияет, но на карте land use это неточность.</li>
      <li><b>Склады справа</b> — <b>промышленность</b>: это подтверждается и OSM, и проверкой территории.</li>
      <li><b>51 квартал без land use</b> — это в основном узкие полосы вдоль дорог
          (4,3 % площади). Они не искажают общую картину, но в таблице у них стоит «Нет данных».</li>
    </ul>
    """


def build_schools_tables(
    schools: gpd.GeoDataFrame,
    blocks_schools: gpd.GeoDataFrame | None,
) -> tuple[str, str]:
    if blocks_schools is not None and "capacity" in blocks_schools.columns:
        per_block = blocks_schools[blocks_schools["capacity"].fillna(0) > 0].copy()
        per_block = per_block.sort_values("capacity", ascending=False)
        rows = [
            [
                f"Кв. {int(row['block_id'])}",
                fmt_num(row.get("count", 0)),
                fmt_num(row.get("capacity", 0)),
            ]
            for _, row in per_block.iterrows()
        ]
        block_table = html_table(
            ["Квартал", "Школ, шт.", "Вместимость (модель), мест"],
            rows,
            numeric_from=1,
        )
    else:
        block_table = '<p class="muted">Файл blocks_schools.geojson не найден.</p>'

    names = sorted(str(name) for name in schools.get("name", []) if not is_blank(name))
    name_rows = [[str(index + 1), esc(name)] for index, name in enumerate(names)]
    name_table = html_table(["№", "Название школы (OSM)"], name_rows)
    return block_table, name_table


def build_top_table(gdf: gpd.GeoDataFrame) -> str:
    top = gdf.sort_values("population", ascending=False).head(15)
    rows: list[list[str]] = []
    for _, row in top.iterrows():
        provision = row.get("provision_strong")
        rows.append(
            [
                f"Кв. {int(row['block_id'])}",
                esc(LAND_USE_RU.get(None if is_blank(row.get("land_use")) else str(row["land_use"]), "Нет данных")),
                fmt_num(row.get("population")),
                fmt_num(row.get("school_capacity")),
                fmt_num(row.get("demand")),
                fmt_pct(provision),
                fmt_num(row.get("fsi"), 2),
                fmt_num(row.get("gsi"), 2),
                fmt_num(row.get("l"), 1),
                fmt_min(row.get("area_accessibility")),
                esc(translate_morph(row.get("spacematrix_morphotype")) or "—"),
            ]
        )
    return html_table(
        [
            "Квартал",
            "Land use",
            "Население, чел.",
            "Вместимость, мест",
            "Спрос, мест",
            "Обеспеченность",
            "FSI",
            "GSI",
            "Этажность",
            "Доступность",
            "Морфотип Spacematrix",
        ],
        rows,
        numeric_from=2,
    )


def build_all_blocks_table(gdf: gpd.GeoDataFrame) -> str:
    ordered = gdf.sort_values("population", ascending=False)
    rows: list[list[str]] = []
    for _, row in ordered.iterrows():
        rows.append(
            [
                f"Кв. {int(row['block_id'])}",
                esc(LAND_USE_RU.get(None if is_blank(row.get("land_use")) else str(row["land_use"]), "Нет данных")),
                fmt_num(row.get("site_area"), 0),
                fmt_num(row.get("population")),
                fmt_num(row.get("population_density_ha"), 1),
                fmt_num(row.get("school_capacity")),
                fmt_num(row.get("demand")),
                fmt_pct(row.get("provision_strong")),
                fmt_num(row.get("fsi"), 2),
                fmt_num(row.get("gsi"), 2),
                fmt_num(row.get("mxi"), 2),
                fmt_num(row.get("l"), 1),
                fmt_num(row.get("osr"), 2),
                fmt_min(row.get("area_accessibility")),
                esc(translate_morph(row.get("spacematrix_morphotype")) or "—"),
            ]
        )
    return html_table(
        [
            "Квартал",
            "Land use",
            "Площадь, м²",
            "Население",
            "Плотность, чел./га",
            "Вместимость",
            "Спрос",
            "Обеспеченность",
            "FSI",
            "GSI",
            "MXI",
            "L",
            "OSR",
            "Доступность",
            "Морфотип",
        ],
        rows,
        table_id="allBlocksTable",
        numeric_from=2,
    )


def build_metric_dictionary() -> str:
    """Словарь полей blocks_results.geojson: что значит, в чём измеряется, откуда взялось."""
    entries: list[tuple[str, str, str, str, str]] = [
        (
            "block_id",
            "Номер квартала",
            "—",
            "модель",
            "Сквозной номер после объединения «пылинок». Совпадает с номером в матрице времени.",
        ),
        (
            "land_use",
            "Функциональная зона",
            "категория",
            "исходные + правило",
            "Доминирующий класс OSM landuse в квартале. Если зоны нет — «Нет данных».",
        ),
        (
            "land_use_share",
            "Доля ведущей зоны",
            "доля 0–1",
            "модель",
            "Какая часть площади квартала занята ведущим классом. Низкая доля — смешанный квартал.",
        ),
        (
            "lu_share_*",
            "Доли зон по классам",
            "доля 0–1",
            "исходные",
            "Семь полей: residential, business, recreation, industrial, transport, special, agriculture.",
        ),
        (
            "site_area",
            "Площадь квартала",
            "м²",
            "модель",
            "Площадь полигона квартала в EPSG:32636 после объединения мелких частей.",
        ),
        (
            "footprint_area",
            "Площадь застройки",
            "м²",
            "исходные",
            "Сумма пятен зданий OSM внутри квартала.",
        ),
        (
            "build_floor_area",
            "Площадь этажей",
            "м²",
            "модель",
            "Пятно × этажность. Пропущенная этажность заменена на 1 этаж.",
        ),
        (
            "living_area",
            "Жилая площадь",
            "м²",
            "модель",
            "Часть площади этажей, отнесённая к жилью по доле жилых помещений.",
        ),
        (
            "share_living",
            "Доля жилой площади",
            "доля 0–1",
            "модель",
            "living_area / build_floor_area. 0 — жилья в квартале нет.",
        ),
        (
            "share_non_living",
            "Доля нежилой площади",
            "доля 0–1",
            "модель",
            "1 − share_living. Используется для морфотипов и MXI.",
        ),
        (
            "buildings_count",
            "Число зданий",
            "шт.",
            "исходные",
            "Сколько зданий OSM попало в квартал. Пустое значение — зданий нет.",
        ),
        (
            "fsi",
            "FSI — коэффициент плотности застройки",
            "м² этажей / м² участка",
            "модель",
            "build_floor_area / site_area. 0 — застройки нет.",
        ),
        (
            "gsi",
            "GSI — коэффициент застроенности",
            "м² пятна / м² участка",
            "модель",
            "footprint_area / site_area. Чем больше, тем меньше открытых дворов.",
        ),
        (
            "mxi",
            "MXI — индекс смешанности",
            "0–1",
            "модель",
            "Насколько в квартале смешаны жильё и другие функции. 1 — максимально смешанный.",
        ),
        (
            "l",
            "L — средняя этажность",
            "этажей",
            "модель",
            "build_floor_area / footprint_area. Пропуски этажности занижают значение.",
        ),
        (
            "osr",
            "OSR — обеспеченность открытыми пространствами",
            "м² / м²",
            "модель",
            "Доля незастроенной площади квартала. Больше — больше свободного места.",
        ),
        (
            "spacematrix_cluster",
            "Кластер Spacematrix",
            "код",
            "модель",
            "Формальный кластер по FSI, GSI и доле жилья. Нужен для морфотипа.",
        ),
        (
            "spacematrix_morphotype",
            "Морфотип Spacematrix",
            "категория",
            "модель",
            "Название типа застройки. Пусто там, где застройки или жилья нет.",
        ),
        (
            "strelka_morphotype",
            "Морфотип Strelka",
            "категория",
            "модель",
            "Альтернативная типология застройки. Названия кластеров условны.",
        ),
        (
            "population",
            "Население",
            "чел.",
            "модель",
            "living_area / 30 м² на человека. Не официальная статистика округа.",
        ),
        (
            "population_density_ha",
            "Плотность населения",
            "чел./га",
            "модель",
            "population / (site_area / 10 000). Считается только для кварталов с жителями.",
        ),
        (
            "school_capacity",
            "Вместимость школ квартала",
            "мест",
            "модель",
            "Число школ × 250 мест. В OSM вместимости нет, поэтому это допущение.",
        ),
        (
            "demand",
            "Спрос на школьные места",
            "мест",
            "модель",
            "population × 120 / 1000. Норматив из конфигурации BlocksNet.",
        ),
        (
            "demand_within",
            "Спрос, покрытый своими местами",
            "мест",
            "модель",
            "min(demand, capacity). BlocksNet сначала отдаёт кварталу его собственные места.",
        ),
        (
            "demand_without",
            "Спрос без покрытия",
            "мест",
            "модель",
            "demand − demand_within. При дефиците мест перераспределения между кварталами нет.",
        ),
        (
            "capacity_within",
            "Использованная вместимость",
            "мест",
            "модель",
            "Сколько своих мест ушло на своё же население.",
        ),
        (
            "capacity_without",
            "Свободная вместимость",
            "мест",
            "модель",
            "capacity − capacity_within. В этом расчёте почти везде 0.",
        ),
        (
            "provision_strong",
            "Обеспеченность (strong)",
            "доля 0–1",
            "модель",
            "Покрытый спрос / весь спрос. Пусто (не 0), если в квартале нет жителей.",
        ),
        (
            "provision_weak",
            "Обеспеченность (weak)",
            "доля 0–1",
            "модель",
            "При max_depth = 1 совпадает с provision_strong — поведение библиотеки.",
        ),
        (
            "area_accessibility",
            "Площадная доступность",
            "минут",
            "модель",
            "Среднее время на автомобиле от квартала до всех кварталов округа.",
        ),
        (
            "geometry",
            "Геометрия квартала",
            "полигон",
            "модель",
            "Полигон в EPSG:4326. На карте упрощён, чтобы файл остался меньше 20 МБ.",
        ),
    ]

    rows = [
        [
            f"<code>{esc(field)}</code>",
            esc(title),
            esc(unit),
            esc(origin),
            esc(explanation),
        ]
        for field, title, unit, origin, explanation in entries
    ]
    table = html_table(
        ["Поле в GeoJSON", "Показатель", "Единица", "Тип", "Что это значит"],
        rows,
    )
    return (
        '<details class="dict">'
        "<summary>Словарь всех 37 полей файла blocks_results.geojson "
        "(нажмите, чтобы раскрыть)</summary>"
        '<p class="small muted" style="margin:10px 0 8px">'
        "«исходные» — пришло из OSM или готового графа; «модель» — оценка BlocksNet / IduEdu; "
        "«исходные + правило» — класс OSM, переклассифицированный по локальному правилу."
        "</p>"
        f'<div class="table-scroll" style="max-height:460px">{table}</div>'
        "</details>"
    )


def build_sources_table(summary: dict[str, object], summary_mtime: float) -> str:
    graph = summary.get("graph", {})
    territory = summary.get("territory", {})
    rows = [
        ["Граница округа", "OpenStreetMap, отношение R1185384", "osmnx.geocode_to_gdf"],
        ["Данные OSM", "OpenStreetMap через Overpass", "зеркало maps.mail.ru"],
        ["Дорожный граф", "BlocksNet Data v0.0.1, graph_drive.pickle", esc(str(graph.get("source", "")))],
        ["Режим передвижения", "автомобиль (drive)", "без пробок и без общественного транспорта"],
        ["Вес рёбер", "time_min", "минуты"],
        ["Обрезка графа", f"буфер {fmt_num(graph.get('buffer_m'))} м", "крупнейшая сильно связная компонента + перенумерация 0…N−1"],
        ["Узлов / рёбер", f"{fmt_num(graph.get('nodes'))} / {fmt_num(graph.get('edges'))}", "после обрезки"],
        ["Библиотека", "BlocksNet 1.0.0a9", "IduEdu"],
        ["Рабочая СК", str(territory.get("work_crs", WORK_CRS)), "метры"],
        ["Население", "модель: жилая площадь / 30 м² на человека", "не официальная статистика"],
        ["Вместимость школ", "модель: 250 мест на школу", "в OSM вместимости нет"],
        ["Норматив", "120 мест на 1000 жителей; 15 минут", "конфигурация BlocksNet"],
        ["Дата файлов", datetime.fromtimestamp(summary_mtime).strftime("%d.%m.%Y %H:%M"), "время последнего расчёта ноутбука"],
    ]
    return html_table(["Что", "Источник / метод", "Примечание"], rows)


def build_limitations() -> str:
    items = [
        "<b>Автомобильный граф вместо пешеходного.</b> Школьники обычно ходят пешком. "
        "15 минут на машине покрывают почти весь округ, поэтому обеспеченность может быть завышена. "
        "Это главное ограничение.",
        "<b>Вместимость школ — модель.</b> В OSM нет числа мест, поэтому всем 19 школам поставлено "
        "минимальное значение 250 мест. Реальные школы Петербурга обычно рассчитаны на 600–1200 учеников. "
        "При 600 местах вместимость была бы 11 400, при 1200 — 22 800 мест, то есть обеспеченность "
        "могла бы составить примерно от 70 % до полного покрытия. Это простая арифметика, а не пересчёт модели.",
        "<b>Обеспеченность считается до перераспределения между кварталами.</b> BlocksNet сначала отдаёт "
        "каждому кварталу его собственные места. Так как мест в целом меньше спроса, вся вместимость "
        "расходуется «на себя», и файл связей «квартал → школа» остаётся пустым. Поэтому 29 % — это "
        "не 15-минутная доступность, а отношение «свои места / свой спрос».",
        "<b>Население — модель, а не статистика.</b> Получилось 136 053 чел., тогда как официальные данные "
        "округа — примерно 115–120 тыс. чел. Модель завышает население на 13–18 %; причина — допущения "
        "о жилой площади и этажности.",
        "<b>Этажность — частично модель.</b> Пропуски заполняются 1 этажом, поэтому население скорее занижается "
        "в одних кварталах и завышается в других.",
        "<b>Функциональные зоны — OSM landuse,</b> а не официальные ПЗЗ и не кадастр.",
        "<b>Граф без пробок и без общественного транспорта.</b> Это не прогноз реального времени поездки.",
        "<b>Правки land use — отклонения от pipeline.ipynb.</b> Парки, парковки, railway/depot → transport, "
        "фильтр узких газонов и объединение «пылинок» меньше 500 м² сделаны по локальной проверке территории.",
        "<b>Границы территории.</b> Всё, что за границей округа, в площадной доступности не учитывается, "
        "хотя жители реально пользуются соседними районами.",
        "<b>Данные OSM могут быть неполными.</b> Школы, здания и дороги могли измениться после даты выгрузки.",
        "<b>Автобусный парк справа остался industrial.</b> В OSM он размечен landuse=industrial, а не depot, "
        "поэтому правило «депо → транспорт» не сработало. На население и школы это не влияет, "
        "но на карте территория выглядит промышленной.",
        "<b>Матрица времени падала без исправления графа.</b> IduEdu использует номера узлов как индексы массива, "
        "а после обрезки граф сохранял номера всего Петербурга. В ячейке 8.2 добавлены выделение крупнейшей "
        "сильно связной компоненты и перенумерация узлов 0…N−1. Без этого шага матрица не считалась.",
        "<b>provision_weak = provision_strong.</b> При глубине 1 BlocksNet учитывает только связи в пределах "
        "15 минут, поэтому «слабый» показатель совпадает с «сильным». Это поведение библиотеки, а не ошибка расчёта.",
    ]
    return "<ol class=\"limits\">" + "".join(f"<li>{item}</li>" for item in items) + "</ol>"


def build_conclusions(summary: dict[str, object]) -> str:
    population = summary.get("population", {})
    schools = summary.get("schools", {})
    travel = summary.get("travel_time", {})
    area = summary.get("territory", {}).get("area_km2")
    return f"""
    <ol class="conclusions">
      <li><b>Компактный округ.</b> Площадь — {fmt_num(area, 2)} км², 97 кварталов.
          Медианная поездка между кварталами на автомобиле — {fmt_num(travel.get('median_min'), 1)} мин,
          90-й процентиль — {fmt_num(travel.get('p90_min'), 1)} мин, недостижимых пар нет.
          Это означает, что на машине округ проезжается почти целиком за 10–15 минут.</li>
      <li><b>Население сконцентрировано.</b> Жители есть только в 13 кварталах из 97;
          модель даёт {fmt_num(population.get('total'))} чел. и плотность
          {fmt_num(population.get('density_per_km2'))} чел./км². Остальные кварталы — парки,
          склады, транспорт, пустыри. Официальные данные округа ниже — около 115–120 тыс. чел.</li>
      <li><b>Школы есть, но вместимость неизвестна.</b> 19 школ, 4 750 модельных мест
          (по 250 на школу), расчётный спрос — {fmt_num(schools.get('demand_total'))} мест.
          Формальная обеспеченность — {fmt_pct(schools.get('provision_strong'))}.</li>
      <li><b>29 % — нижняя граница, а не факт.</b> Если реальные школы рассчитаны на 600–1200 мест,
          обеспеченность была бы заметно выше. Кроме того, при дефиците мест BlocksNet не
          перераспределяет вместимость между кварталами. Поэтому цифру нельзя использовать
          как готовый вывод «школ не хватает».</li>
      <li><b>Проверка территории совпала с картой.</b> Парк Малиновка и парк у Охты — рекреация;
          Хасанский рынок и «Лента» — бизнес; склады справа — промышленность.
          Единственная заметная неточность — автобусный парк, размеченный в OSM как industrial.</li>
      <li><b>Что делать дальше.</b> Собрать реальную вместимость школ (600–1200 мест),
          построить пешеходный граф для школьной доступности, взять официальное население
          и проверить кварталы без land use. После этого 29 % можно заменить на защищаемую оценку.</li>
    </ol>
    """


# ---------------------------------------------------------------------------
# Шаблон страницы
# ---------------------------------------------------------------------------


def page_template() -> str:
    return r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Пороховые — дашборд исследования</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
%%FOLIUM_HEAD%%
<style>
:root{
  --bg:#f3f6fa; --panel:#ffffff; --ink:#16202b; --muted:#5c6b7a;
  --line:#dbe2ea; --accent:#1f4e79; --accent2:#2a9d8f; --warn:#b45309;
  --danger:#b42318; --soft:#eef3f9;
}
*{box-sizing:border-box}
html,body{height:auto;margin:0;padding:0}
body{
  background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:15px; line-height:1.55;
}
a{color:var(--accent)}
.wrap{max-width:1420px;margin:0 auto;padding:0 22px 48px}
header.hero{
  background:linear-gradient(135deg,#102c46 0%,#1f4e79 62%,#2a6f97 100%);
  color:#fff;padding:38px 0 30px;margin-bottom:20px;
}
header.hero .wrap{padding-bottom:0}
.hero h1{margin:0 0 8px;font-size:32px;line-height:1.2;letter-spacing:-.02em}
.hero p{margin:0;color:#d7e4f0;max-width:900px}
.hero .meta{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.chip{background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.22);
  padding:5px 10px;border-radius:999px;font-size:12px;color:#eaf2f9}
nav.toc{position:sticky;top:0;z-index:1200;background:rgba(243,246,250,.95);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);margin-bottom:18px}
nav.toc .wrap{display:flex;gap:6px;flex-wrap:wrap;padding-top:10px;padding-bottom:10px}
[id]{scroll-margin-top:80px}
nav.toc a{font-size:13px;text-decoration:none;color:#33475b;padding:5px 10px;border-radius:8px}
nav.toc a:hover{background:#e3ebf4;color:var(--accent)}
section.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;
  padding:22px 24px;margin:18px 0;box-shadow:0 2px 10px rgba(18,50,79,.05)}
section.panel h2{margin:0 0 6px;font-size:22px;letter-spacing:-.01em}
section.panel h3{margin:22px 0 8px;font-size:17px}
.lead{color:var(--muted);margin:0 0 16px;max-width:1050px}
.muted{color:var(--muted)}
.small{font-size:12.5px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:16px 0 4px}
.kpi{background:linear-gradient(180deg,#fff,#f8fafc);border:1px solid var(--line);
  border-radius:12px;padding:13px 14px}
.kpi-label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.kpi-value{font-size:24px;font-weight:700;line-height:1.25;margin-top:4px;color:#12324f}
.kpi-sub{font-size:11.5px;color:var(--muted);margin-top:3px}
.callout{background:#fbfdff;border:1px solid var(--line);
  border-left:5px solid var(--accent);padding:14px 16px;margin:12px 0;border-radius:10px}
.callout.danger{border-left-color:var(--danger);background:#fff8f7}
.callout.warn{border-left-color:var(--warn);background:#fffbf3}
.callout.ok{border-left-color:var(--accent2);background:#f5fcfb}
.callout h4{margin:0 0 5px;font-size:15px}
.callout p{margin:0}
.callout ul{margin:6px 0 0 18px;padding:0}
.map-frame{height:690px;border:1px solid var(--line);border-radius:12px;overflow:hidden;position:relative;background:#e9eef4}
.map-frame .folium-map{height:100%;width:100%}
.legend-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin-top:14px}
.legend-item{display:flex;gap:9px;align-items:flex-start;font-size:12.5px;color:#3c4d5e;
  background:var(--soft);border-radius:9px;padding:9px 11px}
.swatch{width:16px;height:16px;border-radius:4px;flex:0 0 auto;margin-top:2px;border:1px solid rgba(0,0,0,.08)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-top:14px}
.chart-box{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 14px 10px}
.chart-box h4{margin:0 0 2px;font-size:14px}
.chart-box .cap{font-size:11.5px;color:var(--muted);margin:0 0 8px;min-height:32px}
.chart-canvas{position:relative;height:290px}
.table-scroll{overflow:auto;max-height:560px;border:1px solid var(--line);border-radius:10px}
.tbl{width:100%;border-collapse:collapse;font-size:12.5px}
.tbl th{position:sticky;top:0;background:#eef3f9;color:#25384a;text-align:left;
  padding:9px 10px;border-bottom:1px solid var(--line);font-weight:600;white-space:nowrap;z-index:2}
.tbl td{padding:7px 10px;border-bottom:1px solid #eef1f5;vertical-align:top;overflow-wrap:anywhere}
.tbl tr:nth-child(even) td{background:#fbfcfe}
.tbl td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl tr:hover td{background:#f2f7fd}
.search{width:100%;max-width:420px;padding:9px 12px;border:1px solid var(--line);
  border-radius:9px;font-size:14px;margin:6px 0 12px;outline:none}
.search:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(31,78,121,.1)}
details.dict{border:1px solid var(--line);border-radius:10px;background:#fbfcfe;margin:8px 0 4px}
details.dict>summary{cursor:pointer;padding:11px 14px;font-size:13.5px;font-weight:600;color:#25384a;
  border-radius:10px;list-style:none}
details.dict>summary::-webkit-details-marker{display:none}
details.dict>summary::before{content:"▸ ";color:var(--accent);font-weight:700}
details.dict[open]>summary::before{content:"▾ "}
details.dict>summary:hover{background:#eef3f9}
details.dict>p,details.dict>div{margin-left:14px;margin-right:14px}
details.dict>div{margin-bottom:12px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:900px){.two-col{grid-template-columns:1fr}.charts{grid-template-columns:1fr}.map-frame{height:520px}}
ul.notes,ol.limits,ol.conclusions{margin:8px 0 0 20px;padding:0}
ul.notes li,ol.limits li,ol.conclusions li{margin-bottom:9px}
ol.limits li::marker,ol.conclusions li::marker{font-weight:700;color:var(--accent)}
code{background:#eef3f9;padding:1px 5px;border-radius:4px;font-size:12.5px}
pre{background:#0f2537;color:#e8f0f8;padding:14px 16px;border-radius:10px;overflow:auto;font-size:13px}
footer{color:var(--muted);font-size:12.5px;text-align:center;padding:26px 0 6px}
@media print{nav.toc{display:none}.map-frame{height:420px}body{background:#fff}}
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <h1>Пороховые: кварталы, население, школы, транспорт</h1>
    <p>Интерактивный дашборд учебного исследования по примеру <code>pipeline.ipynb</code>.
       Все показатели уже посчитаны в ноутбуке; этот файл только показывает готовые результаты.</p>
    <div class="meta">
      <span class="chip">Муниципальный округ Пороховые</span>
      <span class="chip">OSM R1185384</span>
      <span class="chip">BlocksNet 1.0.0a9</span>
      <span class="chip">Граф: автомобиль, time_min</span>
      <span class="chip">Собрано: %%BUILD_DATE%%</span>
      <span class="chip">Размер: %%FILE_SIZE%%</span>
    </div>
  </div>
</header>

<nav class="toc"><div class="wrap">
  <a href="#kpi">Главное</a>
  <a href="#warnings">Важно</a>
  <a href="#map-section">Карта</a>
  <a href="#charts">Графики</a>
  <a href="#territory">Территория</a>
  <a href="#tables">Таблицы</a>
  <a href="#schools">Школы</a>
  <a href="#conclusions">Выводы</a>
  <a href="#limits">Ограничения</a>
  <a href="#sources">Источники</a>
  <a href="#rerun">Как пересобрать</a>
</div></nav>

<div class="wrap">

<section class="panel" id="kpi">
  <h2>Главное в цифрах</h2>
  <p class="lead">Это модельные оценки, а не официальная статистика. Ниже — ключевые числа,
     которые нужно читать вместе с ограничениями.</p>
  %%KPI%%
</section>

<section class="panel" id="warnings">
  <h2>Что важно знать до чтения карт</h2>
  <div class="callout danger">
    <h4>Обеспеченность 29 % — артефакт вместимости, а не приговор школам</h4>
    <p>В OSM нет числа мест в школах, поэтому всем 19 школам поставлено минимальное значение
       <b>250 мест</b>. Реальные школы Петербурга обычно рассчитаны на <b>600–1200</b> учеников.
       При 600 местах вместимость была бы 11 400, при 1200 — 22 800 мест. Это дало бы
       ориентировочно <b>от 70 % до полного покрытия спроса</b> (простая арифметика,
       а не пересчёт модели).</p>
    <p style="margin-top:8px">Дополнительно: BlocksNet сначала отдаёт каждому кварталу его собственные места.
       Так как мест меньше спроса, вся вместимость расходуется «на себя», и перераспределения
       между кварталами не остаётся. Поэтому карта обеспеченности показывает не 15-минутную
       доступность, а отношение «свои места / свой спрос».</p>
  </div>
  <div class="callout warn">
    <h4>Население 136 тыс. — модель, официально около 115–120 тыс.</h4>
    <p>Число жителей восстановлено из жилой площади по норме 30 м² на человека.
       Оно завышает официальные данные примерно на 13–18 %. Использовать его можно
       для сравнения кварталов между собой, но не как статистику округа.</p>
  </div>
  <div class="callout">
    <h4>Доступность — на автомобиле, без пробок</h4>
    <p>Граф — готовый <code>graph_drive.pickle</code> из BlocksNet Data, вес <code>time_min</code>,
       время в минутах. Школьники обычно ходят пешком, поэтому 15 минут на машине покрывают
       почти весь округ и обеспеченность может быть завышена.</p>
  </div>
  <div class="callout ok">
    <h4>Граф пришлось исправить, иначе матрица не считалась</h4>
    <p>IduEdu использует номера узлов как индексы массива. После обрезки граф сохранял номера
       всего Петербурга, и ядро аварийно завершалось. В ячейке 8.2 добавлены выделение
       крупнейшей сильно связной компоненты и перенумерация узлов 0…N−1.
       Итог: 347 узлов и 658 рёбер, недостижимых пар нет.</p>
  </div>
</section>

<section class="panel" id="map-section"><!-- не "map": folium задаёт #map {position:absolute} -->
  <h2>Карта кварталов</h2>
  <p class="lead">В правом верхнем углу карты включайте <b>один</b> тематический слой за раз,
     иначе цвета накладываются. Наведите курсор на квартал — появится карточка со всеми показателями.
     Школы отмечены красными точками, граница округа — синей линией.</p>
  <div class="map-frame">%%FOLIUM_BODY%%</div>
  %%LEGEND%%
  <p class="small muted" style="margin-top:12px">Подложка — Esri World Light Gray (данные © OpenStreetMap и др.). Для тайлов нужен интернет.
     Слои: land use, население, обеспеченность, FSI, GSI, площадная доступность,
     морфотипы Spacematrix и Strelka, школы и уличная сеть графа.</p>
</section>

<section class="panel" id="charts">
  <h2>Графики</h2>
  <p class="lead">Графики построены по уже готовым файлам: суммы, доли и гистограммы.
     Наведите курсор на столбец или сектор, чтобы увидеть точное значение.</p>
  <div class="charts">
    <div class="chart-box">
      <h4>Структура площади по land use</h4>
      <p class="cap">Доли площади кварталов, %. «Нет данных» — узкие полосы вдоль дорог (4,3 % площади).</p>
      <div class="chart-canvas"><canvas id="chartLandUse"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Население по кварталам</h4>
      <p class="cap">Жители есть только в 13 кварталах из 97. Остальные — парки, склады, транспорт.</p>
      <div class="chart-canvas"><canvas id="chartPopulation"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Обеспеченность школами по кварталам</h4>
      <p class="cap">11 кварталов со школами и 2 без школ. Красный — ниже 30 %, жёлтый — 30–60 %, зелёный — выше 60 %.</p>
      <div class="chart-canvas"><canvas id="chartProvision"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Вместимость и спрос</h4>
      <p class="cap">Спрос по нормативу 120 мест на 1000 жителей против модельной вместимости 250 мест на школу.</p>
      <div class="chart-canvas"><canvas id="chartCapacity"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Распределение FSI</h4>
      <p class="cap">69 из 97 кварталов не имеют застройки (FSI = 0): парки, склады, транспорт, пустыри.</p>
      <div class="chart-canvas"><canvas id="chartFsi"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Время в пути между кварталами</h4>
      <p class="cap">Все пары кварталов, кроме диагонали (9 312 значений). Медиана — 7,0 мин, 90-й процентиль — 9,8 мин.</p>
      <div class="chart-canvas"><canvas id="chartTravel"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Площадная доступность</h4>
      <p class="cap">Среднее время до всех кварталов округа, минуты. Чем меньше, тем «центральнее» квартал.</p>
      <div class="chart-canvas"><canvas id="chartAccess"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Морфотипы Spacematrix</h4>
      <p class="cap">Только 28 кварталов с застройкой; у остальных морфотип не определяется.</p>
      <div class="chart-canvas"><canvas id="chartMorph"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>Морфотипы Strelka</h4>
      <p class="cap">Альтернативная типология. Названия кластеров условны и зависят от набора кварталов.</p>
      <div class="chart-canvas"><canvas id="chartStrelka"></canvas></div>
    </div>
  </div>
</section>

<section class="panel" id="territory">
  <h2>Проверка территории</h2>
  <p class="lead">Что подтвердилось при визуальной проверке округа и совпало с картой land use.</p>
  %%TERRITORY_NOTES%%
</section>

<section class="panel" id="tables">
  <h2>Таблицы показателей</h2>
  <h3>Паспорт территории</h3>
  %%SUMMARY_TABLE%%
  <h3>Топ-15 кварталов по населению</h3>
  <p class="small muted">Пустая обеспеченность («—») означает, что в квартале нет жителей и спроса;
     это не ноль и не «плохая школа».</p>
  <div class="table-scroll">%%TOP_TABLE%%</div>
  <h3>Все кварталы</h3>
  <p class="small muted">97 строк. Введите номер квартала или класс land use, чтобы отфильтровать таблицу.</p>
  <input class="search" id="blockSearch" type="search" placeholder="Фильтр: номер квартала, land use, морфотип…"
         oninput="filterTable('blockSearch','allBlocksTable')">
  <div class="table-scroll">%%ALL_TABLE%%</div>
  <h3>Словарь показателей</h3>
  <p class="small muted">Что означает каждое поле, в каких единицах оно измеряется и является ли
     оно исходным данным или модельной оценкой.</p>
  %%DICTIONARY_TABLE%%
</section>

<section class="panel" id="schools">
  <h2>Школы</h2>
  <p class="lead">19 школ по данным OSM. Вместимость — модельная: 250 мест на школу,
     потому что в OSM этого поля нет.</p>
  <div class="callout">
    <p><b>Почему нет таблицы связей «квартал → школа».</b> BlocksNet сначала покрывает спрос
       собственными местами квартала. Мест в целом меньше спроса, поэтому вся вместимость
       расходуется «на себя», и файл <code>provision_links.csv</code> остаётся пустым.
       Перераспределения между кварталами в этом расчёте нет.</p>
  </div>
  <div class="two-col">
    <div>
      <h3>Школы по кварталам</h3>
      <div class="table-scroll">%%SCHOOL_BLOCK_TABLE%%</div>
    </div>
    <div>
      <h3>Список школ</h3>
      <div class="table-scroll">%%SCHOOL_LIST%%</div>
    </div>
  </div>
</section>

<section class="panel" id="conclusions">
  <h2>Выводы</h2>
  %%CONCLUSIONS%%
</section>

<section class="panel" id="limits">
  <h2>Ограничения</h2>
  <p class="lead">Эти ограничения нужно учитывать при любом использовании цифр дашборда.</p>
  %%LIMITS%%
</section>

<section class="panel" id="sources">
  <h2>Источники и методика</h2>
  <p class="lead">Отдельно: исходные данные (OSM, готовый граф) и модельные оценки (BlocksNet, IduEdu).</p>
  %%SOURCES_TABLE%%
</section>

<section class="panel" id="rerun">
  <h2>Как пересобрать дашборд</h2>
  <p>Скрипт читает только готовые файлы и ничего не скачивает. Если ноутбук пересчитан,
     запустите его снова — цифры и карта обновятся.</p>
  <pre>cd "C:\Users\azzuh\OneDrive\Dokumen\1. ITMO\ИИ\python-package-boilerplate"
.\.venv\Scripts\python.exe examples\build_porokhovye_dashboard.py</pre>
  <p class="small muted">Результат: <code>examples\dashboard.html</code>. Откройте его двойным щелчком.
     Для карты (тайлы) и графиков (Chart.js) браузеру нужен интернет; сам скрипт — офлайн.</p>
</section>

<footer>
  Учебное исследование муниципального округа Пороховые. Данные OSM и BlocksNet Data,
  расчёты BlocksNet 1.0.0a9. Не является официальной статистикой.
</footer>

</div>

<script>
const CHARTS = %%CHART_DATA%%;
(function () {
  if (window.Chart) {
    Chart.defaults.font.family = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif';
    Chart.defaults.color = '#425466';
    Object.keys(CHARTS).forEach(function (id) {
      const el = document.getElementById(id);
      if (el) { new Chart(el, CHARTS[id]); }
    });
  } else {
    document.querySelectorAll('.chart-canvas').forEach(function (el) {
      el.innerHTML = '<p class="small muted" style="padding:14px">График не загрузился: нет доступа к Chart.js (CDN). Проверьте интернет.</p>';
    });
  }
})();

function filterTable(inputId, tableId) {
  const query = document.getElementById(inputId).value.toLowerCase().trim();
  const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
  rows.forEach(function (row) {
    row.style.display = row.innerText.toLowerCase().indexOf(query) !== -1 ? '' : 'none';
  });
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Сборка страницы
# ---------------------------------------------------------------------------


def build_page(tolerance_m: float) -> str:
    inputs = read_inputs()
    blocks: gpd.GeoDataFrame = inputs["blocks"]  # type: ignore[assignment]
    summary: dict[str, object] = inputs["summary"]  # type: ignore[assignment]
    matrix: pd.DataFrame = inputs["matrix"]  # type: ignore[assignment]
    schools: gpd.GeoDataFrame = inputs["schools"]  # type: ignore[assignment]
    boundary: gpd.GeoDataFrame = inputs["boundary"]  # type: ignore[assignment]
    blocks_schools = inputs["blocks_schools"]
    graph_edges = inputs["graph_edges"]

    gdf = prepare_blocks(blocks, tolerance_m)
    features = build_features(gdf)
    folium_head, folium_body = build_map(gdf, features, boundary, schools, graph_edges)

    charts = chart_configs(gdf, matrix, summary)
    school_block_table, school_list = build_schools_tables(schools, blocks_schools)  # type: ignore[arg-type]

    page = page_template()
    replacements = {
        "%%FOLIUM_HEAD%%": folium_head,
        "%%FOLIUM_BODY%%": folium_body,
        "%%KPI%%": build_kpi(summary, gdf),
        "%%LEGEND%%": build_legend(gdf),
        "%%TERRITORY_NOTES%%": build_territory_notes(),
        "%%SUMMARY_TABLE%%": build_summary_table(summary, gdf),
        "%%TOP_TABLE%%": build_top_table(gdf),
        "%%ALL_TABLE%%": build_all_blocks_table(gdf),
        "%%DICTIONARY_TABLE%%": build_metric_dictionary(),
        "%%SCHOOL_BLOCK_TABLE%%": school_block_table,
        "%%SCHOOL_LIST%%": school_list,
        "%%CONCLUSIONS%%": build_conclusions(summary),
        "%%LIMITS%%": build_limitations(),
        "%%SOURCES_TABLE%%": build_sources_table(summary, float(inputs["summary_mtime"])),
        "%%CHART_DATA%%": json.dumps(charts, ensure_ascii=False),
        "%%BUILD_DATE%%": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)
    return page


def main() -> None:
    if not DATA_DIR.exists():
        fail(f"Не найдена папка с данными: {DATA_DIR}")

    print("Читаю готовые файлы и собираю дашборд…")
    page = ""
    size_mb = 0.0
    for tolerance in SIMPLIFY_TOLERANCES_M:
        page = build_page(tolerance)
        size_mb = len(page.encode("utf-8")) / (1024 * 1024)
        print(f"  упрощение геометрии {tolerance:g} м → {size_mb:.2f} МБ")
        if size_mb <= MAX_SIZE_MB:
            break
    else:
        fail(
            f"Даже при сильном упрощении файл больше {MAX_SIZE_MB:g} МБ. "
            "Уменьшите число слоёв или геометрию."
        )

    page = page.replace("%%FILE_SIZE%%", f"{size_mb:.2f} МБ")
    if "%%" in page:
        fail(
            "В шаблоне остались незаполненные подстановки вида %%NAME%%. "
            "Проверьте функцию page_template() и словарь replacements."
        )
    OUTPUT_PATH.write_text(page, encoding="utf-8")
    final_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print()
    print("=" * 64)
    print(f"[ГОТОВО] {OUTPUT_PATH}")
    print(f"Размер: {final_mb:.2f} МБ (лимит {MAX_SIZE_MB:g} МБ)")
    print("Откройте файл двойным щелчком в браузере.")
    print("=" * 64)


if __name__ == "__main__":
    main()
