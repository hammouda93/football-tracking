from django import template

from matches.services import format_timecode


register = template.Library()


@register.filter
def timecode(value):
    return format_timecode(value)


@register.filter
def metric(metrics, key):
    if isinstance(metrics, dict):
        return metrics.get(key, 0)
    return 0


@register.filter
def percent(value):
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


@register.filter
def number(value):
    try:
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else f"{numeric:.1f}"
    except (TypeError, ValueError):
        return "0"


@register.filter
def status_class(value):
    return {
        "completed": "success",
        "auto_accepted": "success",
        "validated": "success",
        "corrected": "success",
        "processing": "active",
        "queued": "active",
        "review": "warning",
        "pending": "warning",
        "failed": "danger",
        "rejected": "danger",
    }.get(str(value), "neutral")


@register.filter
def percentage_width(value):
    try:
        return round(max(0.0, min(1.0, float(value))) * 100, 1)
    except (TypeError, ValueError):
        return 0
