# combra.viz

```{versionadded} 0.4
```

Shared plotting theme used by every `combra.*` plot module, so figures are visually
consistent and the palette / axis styling / PNG export live in one place instead of
being duplicated per module. Importing it requires the `[viz]` extra (plotly).

```python
from combra import viz
```

````{py:data} combra.viz.PALETTE

The categorical series colour cycle (`['orange', 'red', 'blue', 'green', 'indigo']`).
````

````{py:data} combra.viz.MARKERS

The marker-symbol cycle (`['triangle-down', 'square', 'diamond', ...]`).
````

````{py:data} combra.viz.MARKER_GLYPHS

`{plotly_marker: unicode_glyph}` map used for the per-cell legend annotations.
````

````{py:function} combra.viz.axis_style(**overrides) -> dict

The shared plotly axis styling (black frame + light grid). Returns the styling keys
every combra axis uses; merge caller-specific keys (`title` / `tickvals` / `range` / …)
alongside it.

:param overrides: Extra axis keys merged over the shared style.
:returns: **style** – the axis-styling dict.
:rtype: dict

**Example**

```python
>>> from combra.viz import axis_style
>>> xaxis = dict(title=dict(text='angle'), tickvals=[0, 180, 360], **axis_style())
```
````

````{py:function} combra.viz.export_png(fig, path, scale=2) -> None

Write a plotly ``fig`` to ``path`` as PNG (via kaleido).

:param fig: A plotly figure.
:param path: Output PNG path.
:param scale: Raster scale factor. Default: `2`.
:rtype: None
````
