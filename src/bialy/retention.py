"""Sessions shaped for retention observations."""
import dataclasses


def shape_archetypes(archetypes, settings):
    """Apply the requested follow-up rate and cap."""
    rate, cap = settings.get("follow_up_rate"), settings.get("follow_ups")
    out = []
    for a in archetypes:
        changes = {}
        if rate is not None:
            changes["follow_up_rate"] = float(rate)
        if cap is not None:
            changes["max_follow_ups"] = int(cap)
        out.append(dataclasses.replace(a, **changes) if changes else a)
    return out
