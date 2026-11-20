"""Environment stages by name (train_fault, eval_room, diagnose_room, render_room)."""


def env_class(name):
    if name == "room":
        from room_env import FaultRoom as C
    elif name == "room_dyn":
        from room_dyn_env import FaultRoomDyn as C
    elif name == "terrain":
        from terrain_env import FaultTerrain as C
    elif name == "terrain_dyn":
        from terrain_dyn_env import FaultTerrainDyn as C
    else:
        raise KeyError(name)
    return C


SCENES = ("room", "room_dyn", "terrain", "terrain_dyn")
HAS_MOVERS = ("room_dyn", "terrain_dyn")


def view_scene(name):
    """Which render model (room_view.RoomView scene) shows this stage."""
    return "terrain" if name.startswith("terrain") else "room"
